from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from firecrawl import FirecrawlApp, ScrapeOptions
from datetime import datetime, timezone
from typing import List, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_community.document_loaders import NotionDBLoader
from notion_client import Client
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
    # llm = ChatDeepSeek(model="deepseek-chat")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    exit(1)

def scrape_content(state: AgentState) -> AgentState:
    """Node 1: Scrape content from the web"""
    print("🔍 STEP 1: Scraping content from TLDR.tech...")
    url = "https://tldr.tech/"
    
    try:
        raw_content = extract_tech_news.invoke({"url": url})
        print(f"✅ Successfully scraped {len(raw_content)} pages")
        
        return {
            "raw_content": raw_content,
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Successfully scraped {len(raw_content)} articles from {url}")
            ]
        }
    except Exception as e:
        print(f"❌ Scraping failed: {str(e)}")
        return {
            "raw_content": [],
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Error scraping content: {str(e)}")
            ]
        }

def extract_structured_news(state: AgentState) -> AgentState:
    """Node 2: Extract structured information from scraped content"""
    print("🧠 STEP 2: Using LLM to extract structured articles...")
    raw_content = state.get("raw_content", [])
    
    if not raw_content:
        print("⚠️ No raw content to process")
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
                "publication_date": datetime.now(timezone.utc).isoformat(),
                "author": "Unknown",
                "article_url": "https://tldr.tech/",
                "main_topic": "Technology",
                "image_url": "No Image Url",
                "technologies": [],
                "content": response.content[:500]
            }]
        
        print(f"✅ Successfully extracted {len(extracted_articles)} articles")
        return {
            "extracted_articles": extracted_articles,
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Successfully extracted {len(extracted_articles)} structured articles")
            ]
        }
        
    except Exception as e:
        print(f"❌ Extraction failed: {str(e)}")
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
    print(f"RELEVANCE NODE: Got {len(extracted_article)} articles to analyze")

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
    print(f"CATEGORIZER NODE: Got {len(articles)} articles to analyze")

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
    print(f"SUMMARIZER NODE: Got {len(articles)} articles to analyze")

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

def save_news_to_notion(state: AgentState) -> AgentState:
    """Node 6: Save analyzed articles to Notion database"""
    print("💾 STEP 6: Saving articles to Notion database...")
    articles = state.get("analyzed_articles", [])
    
    if not articles:
        print("⚠️ No articles to save")
        return {
            "messages": state.get("messages", []) + [
                SystemMessage(content="No articles to save to Notion")
            ]
        }
    
    # Helper function to safely get text values
    def safe_text(text_value, default="Unknown"):
        """Safely convert value to string, handling None/null values"""
        if text_value is None:
            return default
        return str(text_value).strip() if str(text_value).strip() else default
    
    try:
        # Initialize Notion client
        notion = Client(auth=os.getenv("NOTION_TOKEN"))
        database_id = os.getenv("DATABASE_ID") or os.getenv("NEWS_ARTICLE_DATABASE_ID")
        
        if not database_id:
            raise ValueError("Missing DATABASE_ID in environment variables")
            
        saved_count = 0
        print(f"📝 Attempting to save {len(articles)} articles...")
        
        for article in articles:
            try:
                print(f"🔄 Processing article: {article.get('title', 'Unknown')[:50]}...")
                
                # Build properties safely
                properties = {
                    "Title": {
                        "title": [{"text": {"content": safe_text(article.get("title"), "Untitled")}}]
                    },
                    "URL": {
                        "url": article.get("article_url", "") or ""
                    },
                    "Source": {
                        "rich_text": [{"text": {"content": safe_text(article.get("source"))}}]
                    },
                    "Author": {
                        "rich_text": [{"text": {"content": safe_text(article.get("author"))}}]
                    },
                    "Summary": {
                        "rich_text": [{"text": {"content": safe_text(article.get("summary"), "")[:1900]}}]
                    },
                    "Content": {
                        "rich_text": [{"text": {"content": safe_text(article.get("content"), "")[:1900]}}]
                    },
                    "Main Topic": {
                        "rich_text": [{"text": {"content": safe_text(article.get("main_topic"))}}]
                    },
                    "Relevance Analysis": {
                        "rich_text": [{"text": {"content": safe_text(article.get("relevance_analysis"), "")[:1900]}}]
                    },
                    # Add default values for required fields
                    "Relevance Score": {
                        "number": 5
                    },
                    "Technical Depth": {
                        "select": {"name": "Intermediate"}
                    },
                    "Post Angle": {
                        "select": {"name": "News Commentary"}
                    },
                    "Engagement Potential": {
                        "select": {"name": "Medium"}
                    },
                    "Status": {
                        "status": {"name": "Analyzed"}
                    },
                    "Trending Potential": {
                        "checkbox": False
                    },
                    "Post Created": {
                        "checkbox": False
                    },
                    "Content Category": {
                        "multi_select": [{"name": "Industry Trends"}]
                    },
                    "Target Audience": {
                        "multi_select": [
                            {"name": "Junior Developers"},
                            {"name": "Mid-level Developers"}
                        ]
                    },
                    "Scraped Date": {
                        "date": {"start": datetime.now().date().isoformat()}
                    }
                }
                
                # Handle optional fields safely
                
                # Image URL
                image_url = article.get("image_url")
                if image_url and isinstance(image_url, str) and image_url.startswith("http"):
                    properties["Image URL"] = {"url": image_url}
                
                # Technologies
                technologies = article.get("technologies", [])
                if technologies and isinstance(technologies, list):
                    clean_techs = [str(tech).strip() for tech in technologies if tech and str(tech).strip()][:10]
                    if clean_techs:
                        properties["Technologies"] = {
                            "multi_select": [{"name": tech[:100]} for tech in clean_techs]
                        }
                
                # Publication Date
                pub_date = article.get("publication_date")
                if pub_date and str(pub_date).strip():
                    try:
                        date_str = str(pub_date).strip()
                        if 'T' in date_str:
                            date_str = date_str.split('T')[0]
                        if len(date_str) >= 10:  # Basic validation
                            properties["Published Date"] = {
                                "date": {"start": date_str}
                            }
                    except Exception as date_error:
                        print(f"⚠️ Date parsing failed for {pub_date}: {date_error}")
                
                # Create the database entry
                notion.pages.create(
                    parent={"database_id": database_id},
                    properties=properties
                )
                
                saved_count += 1
                print(f"✅ Saved: {article.get('title', 'Unknown')[:50]}...")
                
            except Exception as article_error:
                print(f"❌ Failed to save article '{article.get('title', 'Unknown')}': {article_error}")
                continue
        
        print(f"✅ Successfully saved {saved_count} of {len(articles)} articles to Notion!")
        
        return {
            "analyzed_articles": articles,
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Successfully saved {saved_count} of {len(articles)} articles to Notion")
            ]
        }
        
    except Exception as e:
        print(f"❌ Notion saving failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "analyzed_articles": articles,
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Error saving to Notion: {str(e)}")
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
graph.add_node("notion_saver", save_news_to_notion)

# Define the flow - FIXED VERSION
graph.add_edge(START, "scraper")
graph.add_edge("scraper", "extractor") 
graph.add_edge("extractor", "relevance_analyst")
graph.add_edge("relevance_analyst", "categorizer")
graph.add_edge("categorizer", "summarizer")  # ← THIS WAS MISSING!
graph.add_edge("summarizer", "notion_saver")
graph.add_edge("notion_saver", END)

# Compile the graph
app = graph.compile()

# Enhanced run function with better progress indicators

def run_news_analysis():
    """Run the complete news analysis pipeline with detailed debugging"""
    print("🚀 Starting LinkedIn Content Analysis Pipeline...")
    print("=" * 50)
    
    initial_state = {
        "messages": [],
        "raw_content": [],
        "extracted_articles": [],
        "analyzed_articles": [],
    }
    
    try:
        result = app.invoke(initial_state)
        
        print("\n" + "=" * 50)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"📊 Total Messages: {len(result['messages'])}")
        print(f"📰 Articles Processed: {len(result.get('analyzed_articles', []))}")
        
        # Show complete JSON content for debugging
        for i, article in enumerate(result.get('analyzed_articles', []), 1):
            print(f"\n" + "="*60)
            print(f"📄 ARTICLE {i} - COMPLETE JSON:")
            print("="*60)
            
            # Pretty print the entire article JSON
            import json
            try:
                formatted_json = json.dumps(article, indent=2, ensure_ascii=False)
                print(formatted_json)
            except Exception as e:
                print(f"JSON formatting error: {e}")
                print("Raw article data:", article)
            
            print("="*60)
        
        return result
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_news_analysis()