from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from firecrawl import FirecrawlApp, ScrapeOptions
from datetime import datetime, timezone
from typing import List, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from prompt import news_extractor, news_summarizer, news_relevance_score, article_categorizer
import os
import json

load_dotenv()

class news_entry_structure(BaseModel):
    title: str
    article_url: str
    source: str
    publication_date: str
    summary: str
    author: str
    main_topic: str
    technologies: List[str]
    content: str

class news_refined_structure(BaseModel):
    title: str
    url: str
    source: str
    published_date: datetime
    scraped_date: datetime
    summary: str
    key_points: List[str]
    relevance_score: int  # 1-10
    content_category: List[str]
    technical_depth: str
    post_angle: str
    target_audience: List[str]
    engagement_potential: str
    related_technologies: List[str]
    industry_impact: str
    trending_potential: bool
    personal_connection: str
    status: str = "Refined"
    

class AgentState(TypedDict):
    messages: List[BaseMessage]
    raw_content: List[dict]
    extracted_articles: List[dict]
    analyzed_articles: List[dict]



@tool
def extract_tech_news(url: str, limit: int = 2) -> List[dict]:
    """Extract news using official FireCrawl Python SDK"""
    try:
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError("FIRECRAWL_API_KEY environment variable is not set.")
        
        # Initialize the FireCrawl app
        app = FirecrawlApp(api_key=api_key)
        
        # Simple scrape without options first
        result = app.scrape_url(url)
        
        print(f"FireCrawl result: {result}")  # Debug what we get back
        
        # Return the result as-is for now to see the structure
        if result:
            return [result]  # Wrap in list to match expected format
        
        return []
        
    except Exception as e:
        print(f"Error extracting news data: {e}")
        return []


# Initialize LLM
try:
    llm = ChatOpenAI(
        model="deepseek/deepseek-r1-0528:free",  # OpenRouter DeepSeek model name
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),  # Your OpenRouter API key
        openai_api_base="https://openrouter.ai/api/v1",  # OpenRouter base URL
        temperature=0.3
    )
except Exception as e:
    print(f"Error initializing LLM: {e}")
    exit(1)

def scrape_content(state: AgentState) -> AgentState:
    """Node 1: Scrape content from the web"""
    url = "https://tldr.tech/"
    
    try:
        raw_content = extract_tech_news.invoke({"url": url})
        
        return {
            "raw_content": raw_content,
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Successfully scraped {len(raw_content)} articles from {url}")
            ]
        }
    except Exception as e:
        return {
            "raw_content": [],
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Error scraping content: {str(e)}")
            ]
        }

def extract_structured_news(state: AgentState) -> AgentState:
    """Node 2: Extract structured information from scraped content"""
    raw_content = state.get("raw_content", [])
    
    if not raw_content:
        return {
            "extracted_articles": [],
            "messages": state.get("messages", []) + [
                SystemMessage(content="No raw content to process")
            ]
        }
    
    # Extract markdown content from ScrapeResponse object
    if raw_content and len(raw_content) > 0:
        scrape_response = raw_content[0]
        if hasattr(scrape_response, 'markdown') and scrape_response.markdown:
            scraped_content = scrape_response.markdown
        else:
            scraped_content = str(scrape_response)
    else:
        scraped_content = str(raw_content)
    
    # Create the extraction prompt
    system_prompt = news_extractor(
        scraped_content=scraped_content,
        source_url="https://tldr.tech/", 
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    
    # Create messages for the LLM
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Please extract and structure the news articles from the scraped content above.")
    ]
    
    try:
        # Get LLM response
        response = llm.invoke(messages)
        
        # Parse JSON response - handle markdown code blocks
        try:
            response_text = response.content.strip()
            
            # Extract JSON from markdown code blocks if present
            if "```json" in response_text:
                start_idx = response_text.find("```json") + 7
                end_idx = response_text.find("```", start_idx)
                if end_idx > start_idx:
                    json_text = response_text[start_idx:end_idx].strip()
                else:
                    json_text = response_text[start_idx:].strip()
            elif "```" in response_text:
                start_idx = response_text.find("```") + 3
                end_idx = response_text.find("```", start_idx)
                if end_idx > start_idx:
                    json_text = response_text[start_idx:end_idx].strip()
                else:
                    json_text = response_text[start_idx:].strip()
            else:
                # No code blocks, try to find JSON array/object
                json_text = response_text
                for start_char in ['[', '{']:
                    if start_char in json_text:
                        start_idx = json_text.find(start_char)
                        json_text = json_text[start_idx:]
                        break
            
            # Parse the extracted JSON
            extracted_articles = json.loads(json_text)
            if not isinstance(extracted_articles, list):
                extracted_articles = [extracted_articles]
            
        except json.JSONDecodeError as json_error:
            # If JSON parsing fails, create a fallback structure
            extracted_articles = [{
                "title": "Extraction Error - JSON Parse Failed",
                "summary": response.content[:200] + "...",
                "publication_date": datetime.now(datetime.UTC).isoformat(),
                "author": "Unknown",
                "article_url": "https://tldr.tech/",
                "main_topic": "Technology",
                "technologies": [],
                "content": response.content[:500]
            }]
        
        return {
            "extracted_articles": extracted_articles,
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Successfully extracted {len(extracted_articles)} structured articles")
            ]
        }
        
    except Exception as e:
        return {
            "extracted_articles": [],
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Error in news extraction: {str(e)}")
            ]
        }
    
def analyze_relevance(state: AgentState) -> AgentState:
    """"Node 3: This node analyze the relavance of each article"""
    extracted_article =  state.get("extracted_articles", [])
    updated_article= []

    if not extracted_article:
        return {
            "analyzed_articles": [],
            "messages": state.get("messages", []) + [
                SystemMessage(content="No extracted articles to analyze")
            ]
        }

    for article in extracted_article:
        system_prompt = news_relevance_score(
            article=article,
            user_expertise=["software development", "Ai/Ml", "Python development", "Ai agent", "Javascript/React", "Next js", "Web development", "Saas Apps"],
            target_audience=["Junior/Mid-level developers", "Startup Founders", "Freelance developers", "Tech leads"]
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Please analyze the relevance of each article based on the user expertise and target audience.")
        ]

        try:
            response = llm.invoke(messages)

            article_copy = article.copy()
            article_copy["relevance_analysis"] = response.content
            updated_article.append(article_copy)

        except Exception as e:
            print(f"Skipping article due to error {e}")
            continue

    return {
        "analyzed_articles": updated_article,
        "messages": state.get("messages", []) + [
            SystemMessage(content=f"Successfully analyzed {len(updated_article)} articles")
        ]
    }

def categorize_articles(state: AgentState) -> AgentState:
    """Node 4: Categorize each article based on it's content"""
    articles = state.get("analyzed_articles", [])
    updated_article = []
    if not articles:
        return {
            "messages": state.get("messages", []) + [
                SystemMessage(content="No articles to categorize")
            ]
        }

    for article in articles:
        system_prompt = article_categorizer(
            title=article["title"],
            content=article["content"],
            source=article["article_url"]
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Please categorize the article based on it's content.")
        ]

        try:
            response = llm.invoke(messages)

            article_copy = article.copy()
            article_copy["category"] = response.content
            updated_article.append(article_copy)

        except Exception as e:
            print(f"Skipping article due to error {e}")
            continue
    return {
        "analyzed_articles": updated_article,
        "messages": state.get("messages", []) + [
            SystemMessage(content=f"Successfully categorized {len(updated_article)} articles")
        ]
    }

def summarize_content(state: AgentState) -> AgentState:
    """Node 5: Sumazize the content of each article"""
    articles = state.get("analyzed_articles", [])
    updated_article = []
    if not articles:
        return {
            "messages": state.get("messages", []) + [
                SystemMessage(content="No articles to summarize")
            ]
        }

    for article in articles:
        system_prompt = news_summarizer(
            article=article,
            source=article["article_url"],
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Summarize the article based on it's content")
        ]

        try:
            response = llm.invoke(messages)

            article_copy = article.copy()
            article_copy["summary"] = response.content
            updated_article.append(article_copy)
        except Exception as e:
            print(f"Skipping article due to error {e}")
            continue
    
    
    return {
        "analyzed_articles": updated_article,
        "messages": state.get("messages", []) + [
            SystemMessage(content=f"Successfully summarized {len(updated_article)} articles")
        ]
    }

        
# Build the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("scraper", scrape_content)
graph.add_node("extractor", extract_structured_news)
graph.add_node("relevance_analyst", analyze_relevance)
graph.add_node("categorizer", categorize_articles)
graph.add_node("summarizer", summarize_content)

# Define the flow
graph.add_edge(START, "scraper")
graph.add_edge("scraper", "extractor") 
graph.add_edge("extractor", "relevance_analyst")
graph.add_edge("relevance_analyst", "categorizer")
graph.add_edge("categorizer", "summarizer")
graph.add_edge("summarizer", END)

# Compile the graph
app = graph.compile()

# Test function
def run_news_analysis():
    """Run the complete news analysis pipeline"""
    initial_state = {
        "messages": [],
        "raw_content": [],
        "extracted_articles": [],
        "analyzed_articles": [],
    }
    
    result = app.invoke(initial_state)
    
    print("=== Analysis RESULTS ===")
    print(f"Messages: {len(result['messages'])}")
    print(f"analyzed Articles: {len(result.get('analyzed_articles', []))}")
    
    for i, article in enumerate(result.get('analyzed_articles', []), 1):
        print(f"\n--- Article {i} ---")
        print(f"Title: {article.get('title', 'N/A')}")
        print(f"Summary: {article.get('summary', 'N/A')}")
        print(f"Relevance analysis: {article.get('relevance_analysis', [])}")
        print(f"Categories: {article.get('category', [])}")
        print(f"Article url: {article.get('article_url', [])}")
    
    return result

if __name__ == "__main__":
    run_news_analysis()