from datetime import datetime, timedelta
import pandas as pd
import io
import matplotlib.pyplot as plt
from fpdf import FPDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
import aiohttp
import asyncio
import openai
import pymongo
import logging
import sys

# Initialize MongoDB client and OpenAI API key
client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['economic_analysis_db']
results_collection = db['results']
logs_collection = db['logs']
analysis_collection = db['analysis']

openai.api_key = 'your_openai_api_key_here'

# Setup logging
logging.basicConfig(filename='api.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def fetch_news(session, keyword, api_key, num_articles=5):
    url = f"https://newsapi.org/v2/everything?q={keyword}&apiKey={api_key}"
    async with session.get(url) as response:
        response.raise_for_status()
        data = await response.json()
        articles = data.get('articles', [])[:num_articles]
        return [{
            'title': article['title'],
            'description': article['description'],
            'url': article['url'],
            'keyword': keyword
        } for article in articles]

async def gather_economic_data(country_code, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='M')
    gdp_data = pd.DataFrame({'Date': date_range, 'GDP': ['21T'] * len(date_range)})
    inflation_data = pd.DataFrame({'Date': date_range, 'Inflation Rate': ['3.0%'] * len(date_range)})
    unemployment_data = pd.DataFrame({'Date': date_range, 'Unemployment Rate': ['5.0%'] * len(date_range)})
    interest_rate_data = pd.DataFrame({'Date': date_range, 'Interest Rate': ['1.5%'] * len(date_range)})
    debt_data = pd.DataFrame({'Date': date_range, 'Debt to GDP': ['120%'] * len(date_range)})

    return {
        'GDP': gdp_data.to_dict(orient='records'),
        'Inflation': inflation_data.to_dict(orient='records'),
        'Unemployment': unemployment_data.to_dict(orient='records'),
        'Interest Rate': interest_rate_data.to_dict(orient='records'),
        'Debt': debt_data.to_dict(orient='records')
    }

async def get_targeted_news_data(keywords, api_key, num_articles=5):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_news(session, keyword, api_key, num_articles) for keyword in keywords]
        news_data_lists = await asyncio.gather(*tasks)
        return [item for sublist in news_data_lists for item in sublist]

def generate_prompt(economic_data, news_data):
    prompt = "Analyze the following economic data and targeted news to provide insights:\n\n"

    for key, data in economic_data.items():
        df = pd.DataFrame(data)
        if not df.empty:
            prompt += f"**{key} Data:**\n"
            prompt += df.to_string(index=False) + "\n\n"

    prompt += "### Targeted News Data:\n"
    for article in news_data:
        prompt += f"- **Keyword:** {article['keyword']}\n"
        prompt += f"  **Title:** {article['title']}\n"
        prompt += f"  **Description:** {article['description']}\n"
        prompt += f"  **URL:** {article['url']}\n\n"

    prompt += "Provide an analysis of the data and suggest potential impacts on the economy."

    return prompt

def analyze_data_with_llm(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=3000,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except openai.OpenAIError as e:
        logging.error(f"OpenAI API error: {e}")
        return "Error in LLM analysis."

def gather_data(country_code, years):
    try:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=years * 365)
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        keywords = ["banking", "finance", "stock deals", "government deals", "global crimes"]
        api_key = 'your_newsapi_key_here'

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        economic_data = loop.run_until_complete(gather_economic_data(country_code, start_date_str, end_date_str))
        news_data = loop.run_until_complete(get_targeted_news_data(keywords, api_key))

        prompt = generate_prompt(economic_data, news_data)
        analysis = analyze_data_with_llm(prompt)

        results_document = {
            'timestamp': datetime.now(),
            'start_date': start_date_str,
            'end_date': end_date_str,
            'economic_data': economic_data,
            'news_data': news_data,
            'analysis': analysis
        }
        results_collection.insert_one(results_document)

        logging.info('Data gathered and analyzed successfully.')
    
    except Exception as e:
        logging.error(f"Error in gather_data: {e}")
        logs_collection.insert_one({
            'timestamp': datetime.now(),
            'type': 'error',
            'message': f"Error in gather_data: {e}"
        })

def reanalyze_data_process():
    try:
        results_cursor = results_collection.find()
        economic_data = {}
        news_data = []

        for result in results_cursor:
            for key, df in result.get('economic_data', {}).items():
                if key not in economic_data:
                    economic_data[key] = pd.DataFrame(df)
                else:
                    economic_data[key] = pd.concat([economic_data[key], pd.DataFrame(df)], ignore_index=True)
            
            news_data.extend(result.get('news_data', []))

        prompt = generate_prompt(economic_data, news_data)
        analysis = analyze_data_with_llm(prompt)

        analysis_document = {
            'timestamp': datetime.now(),
            'analysis': analysis
        }
        analysis_collection.insert_one(analysis_document)

        logging.info('Reanalysis completed successfully.')

    except Exception as e:
        logging.error(f"Error in reanalyze_data_process: {e}")
        logs_collection.insert_one({
            'timestamp': datetime.now(),
            'type': 'error',
            'message': f"Error in reanalyze_data_process: {e}"
        })

def generate_short_pdf(summary_text):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Economic Data Analysis Report - Short Summary', 0, 1, 'C')
        pdf.ln(10)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 10, summary_text)

        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        pdf_output.seek(0)

        with open('short_report.pdf', 'wb') as f:
            f.write(pdf_output.read())

        logging.info('Short PDF generated successfully.')

    except Exception as e:
        logging.error(f"Error in generate_short_pdf: {e}")
        logs_collection.insert_one({
            'timestamp': datetime.now(),
            'type': 'error',
            'message': f"Error in generate_short_pdf: {e}"
        })

def generate_long_pdf(economic_data, graphs):
    try:
        doc = SimpleDocTemplate("long_report.pdf", pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("Economic Data Analysis Report - Detailed Report", styles['Title']))
        story.append(Paragraph("This report provides detailed insights, including graphs and tables.", styles['BodyText']))
        story.append(Paragraph("<br/><br/>", styles['BodyText']))

        for title, df in economic_data.items():
            story.append(Paragraph(title, styles['Heading2']))
            if df:
                df = pd.DataFrame(df)
                table_data = [df.columns.tolist()] + df.values.tolist()
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))
                story.append(table)
                story.append(Paragraph("<br/><br/>", styles['BodyText']))

        doc.build(story)
        logging.info('Long PDF generated successfully.')

    except Exception as e:
        logging.error(f"Error in generate_long_pdf: {e}")
        logs_collection.insert_one({
            'timestamp': datetime.now(),
            'type': 'error',
            'message': f"Error in generate_long_pdf: {e}"
        })

if __name__ == '__main__':
    step = sys.argv[1]

    if step == 'gather_data':
        country_code = sys.argv[2]
        years = int(sys.argv[3])
        gather_data(country_code, years)
    elif step == 'reanalyze_data':
        reanalyze_data_process()
    elif step == 'generate_short_pdf':
        summary_text = 'Generated summary text'
        generate_short_pdf(summary_text)
    elif step == 'generate_long_pdf':
        economic_data = {'data': 'sample'}  # Example data
        graphs = {}
        generate_long_pdf(economic_data, graphs)
    else:
        print(f'Unknown step: {step}', file=sys.stderr)
        sys.exit(1)
