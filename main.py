from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from firecrawl import FirecrawlApp, ScrapeOptions
from datetime import datetime, timezone, timedelta
from typing import List, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_community.document_loaders import NotionDBLoader
from notion_client import Client
from github import Github, UnknownObjectException, RateLimitExceededException
from repo import analyze_single_repo
from prompt import news_extractor, news_summarizer, news_relevance_score, article_categorizer, repo_significance_analyzer, content_strategist, content_reviewer, linkedin_post_writer, post_variation_generator, engagement_predictor, posting_timing_optimizer
import os
import json, re
import time
import traceback
import random
from openai import RateLimitError

load_dotenv()
    

class AgentState(TypedDict):
    messages: List[BaseMessage]
    raw_content: List[dict]
    extracted_articles: List[dict]
    analyzed_articles: List[dict]
    github_data: List[dict] 
    analyzed_content: List[dict]
    linkedin_posts: List[dict]
    reviewed_posts: List[dict] 
    scheduled_posts: List[dict] 

def robust_json_parse(response_text):
    """Robustly parse JSON from LLM response, handling various formats"""
    if not response_text:
        return None
    
    # Method 1: Direct parsing
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass
    
    # Method 2: Remove markdown code blocks
    try:
        clean_text = re.sub(r'```[a-zA-Z]*\n?', '', response_text)
        clean_text = re.sub(r'\n```', '', clean_text)
        return json.loads(clean_text.strip())
    except json.JSONDecodeError:
        pass
    
    # Method 3: Find JSON object/array in text
    try:
        # Find JSON object
        start_obj = response_text.find('{')
        if start_obj >= 0:
            brace_count = 0
            for i, char in enumerate(response_text[start_obj:], start_obj):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_text = response_text[start_obj:i+1]
                        return json.loads(json_text)
        
        # Find JSON array
        start_arr = response_text.find('[')
        if start_arr >= 0:
            bracket_count = 0
            for i, char in enumerate(response_text[start_arr:], start_arr):
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        json_text = response_text[start_arr:i+1]
                        return json.loads(json_text)
    except json.JSONDecodeError:
        pass
    
    # Method 4: Try to extract JSON between first { and last }
    try:
        first_brace = response_text.find('{')
        last_brace = response_text.rfind('}')
        if first_brace >= 0 and last_brace > first_brace:
            json_text = response_text[first_brace:last_brace+1]
            return json.loads(json_text)
    except json.JSONDecodeError:
        pass
    
    return None

def safe_llm_call(messages, max_retries=3, base_delay=2):
    """Safely call LLM with rate limiting handling and retries"""
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            return response
        
        except Exception as e:
            error_str = str(e).lower()
            
            # Handle rate limiting
            if "429" in error_str or "rate" in error_str:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(1, 3)
                    print(f"⚠️ Rate limited, waiting {delay:.1f}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ Max retries reached due to rate limiting")
                    raise e
            
            # Handle other errors
            else:
                if attempt < max_retries - 1:
                    delay = base_delay + random.uniform(0.5, 1.5)
                    print(f"⚠️ LLM error, retrying in {delay:.1f}s... ({attempt + 2}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ Max retries reached due to: {e}")
                    raise e
    
    raise Exception("Failed after all retries")


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

@tool
def github_repo_monitor():
    """Hybrid approach: Monitor priority repos + discover active ones"""

    g = Github(os.getenv("GITHUB_TOKEN"))
    user = g.get_user()

    priority_repos = [
        "LinkedIn-Content-System",
        "ai-resume-analyzer",
        "Saas-app",
        "portfolio",
        "mini-zentry-design",
    ]

    all_analysis = []

    for repo_name in priority_repos:
        try:
            repo = user.get_repo(repo_name)

            analysis = analyze_single_repo(repo)
            if analysis:
                analysis['priority'] = True
                all_analysis.append(analysis)
                print(f"Priority name '{repo_name}' analyzed")
        
        except Exception as e:
            print(f"❌ Priority repo {repo_name} failed: {e}")


    try:
        recent_repos = user.get_repos(type='owner', sort='updated')

        for repo in recent_repos:
            rate_limit_info = g.get_rate_limit()
            remaining = rate_limit_info.resources.core.remaining
            reset_time = rate_limit_info.resources.core.reset  # tz-aware datetime (UTC)

            if remaining < 10:  # threshold
                # use UTC to avoid naive/aware mismatch
                now = datetime.now(timezone.utc)
                sleep_duration = (reset_time - now).total_seconds() + 5
                print(
                    f"⚠️ Low rate limit ({remaining} calls remaining). "
                    f"Sleeping for {sleep_duration:.0f}s until {reset_time.strftime('%H:%M:%S')}"
                )
                time.sleep(max(0, sleep_duration))

            # Then continue with repo discovery...
            if repo.name in priority_repos or repo.fork or repo.archived or len(all_analysis) >= 8:
                continue

            cutoff_date = datetime.now(timezone.utc) - timedelta(days=14)
            if repo.pushed_at and repo.pushed_at.replace(tzinfo=timezone.utc) > cutoff_date:
                analysis = analyze_single_repo(repo)
                if analysis:
                    analysis['priority'] = False
                    all_analysis.append(analysis)
                    print(f"Discovered active repo '{repo.name}'")

    except RateLimitExceededException:
        print(f"❌ Rate limit exceeded during auto-discovery. Halting process.")
    except Exception as e:
        print(f"❌ Auto-discovery failed with an unexpected error: {e}")



    result = {
        'repositories': all_analysis,
        'priority_repos_count': len([a for a in all_analysis if a.get('priority')]),
        'discovered_repos_count': len([a for a in all_analysis if not a.get('priority')]),
        'total_repos_analyzed': len(all_analysis),
        'last_checked': datetime.now().isoformat(),
        'summary': f"Analyzed {len(all_analysis)} repositories with recent activity"
    }
    
    print(f"🎉 GitHub monitoring complete! Found {len(all_analysis)} active repositories")
    return result

def analyze_github_repos(state: AgentState) -> AgentState:
    """Node 7: This node analyzes the github repos data"""
    print("🔍 Entering analyze_github_repos...")

    try:
        github_result = github_repo_monitor.invoke({})  
        print("📦 github_result keys:", list(github_result.keys()))
    except Exception as e:
        print(f"❌ github_repo_monitor.invoke failed: {e}")
        traceback.print_exc()
        return state

    repositories = github_result.get("repositories", [])
    print(f"📂 Retrieved {len(repositories)} repositories from monitor")

    if not repositories:
        print("⚠️ No github repositories to process")
        return {
            "github_data": [],
            "messages": state.get("messages", []) + [
                SystemMessage(content="No github repositories to process")
            ]
        }

    print(f"🧠 STEP 7: Analyzing {len(repositories)} github repositories...")
    analyzed_repos = []

    for i, repo_data in enumerate(repositories, 1):
        print(f"\n🔄 [{i}/{len(repositories)}] Processing repo: {repo_data.get('name', 'unknown')}")
        try:
            # Debug raw repo_data
            print("   ↳ Repo data keys:", list(repo_data.keys()))

            # Extract repo details
            repo_name = repo_data['name']
            commit_messages = [commit['message'] for commit in repo_data.get('recent_commits', [])]
            content = repo_data.get('content_summary', '')
            diff_summary = f"{repo_data.get('commits_count', 0)} commits, +{repo_data.get('total_additions', 0)} -{repo_data.get('total_deletions', 0)} lines"
            repo_description = repo_data.get('description', '')
            tech_stack = [repo_data.get('language')] if repo_data.get('language') and repo_data.get('language') != 'Unknown' else []
            
            # Build prompt
            system_prompt = repo_significance_analyzer(
                repo_name=repo_name,
                commit_messages=commit_messages,
                repo_content_summary=content,
                diff_summary=diff_summary,
                repo_description=repo_description,
                tech_stack=tech_stack
            )
            print("   ✅ Prompt built successfully")

            # Call LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content="Please analyze this repository for LinkedIn content potential.")
            ]
            
            response = llm.invoke(messages)
            print("   ✅ LLM invoked successfully")

            clean_content = re.sub(r"```(?:json)?|```", "", response.content).strip()
            parsed = json.loads(clean_content)

            if isinstance(parsed, dict):
                analysis = parsed
            elif isinstance(parsed, list) and parsed:
                analysis = parsed[0]
            else:
                raise ValueError(f"Unexpected JSON structure: {parsed}")


            # Merge analysis into repo_data
            repo_copy = repo_data.copy()
            repo_copy.update({"analysis": analysis})
            analyzed_repos.append(repo_copy)
            
            print(f"✅ Successfully analyzed repository: {repo_name}")
            
        except Exception as e:
            print(f"❌ Failed to analyze repo {repo_data.get('name', 'unknown')}: {e}")
            traceback.print_exc()  # <- full error + stack trace
            continue
    
    return {
        "github_data": analyzed_repos,
        "messages": state.get("messages", []) + [
            SystemMessage(content=f"Successfully analyzed {len(analyzed_repos)} repositories")
        ]
    }      


# Initialize LLM
try:
    # llm = ChatOpenAI(
    #     model="deepseek/deepseek-r1-0528:free",  # OpenRouter DeepSeek model name
    #     openai_api_key=os.getenv("OPENROUTER_API_KEY"),  # Your OpenRouter API key
    #     openai_api_base="https://openrouter.ai/api/v1",  # OpenRouter base URL
    #     temperature=0.3
    # )
    llm = ChatDeepSeek(model="deepseek-chat")
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
    """Node 2: Extract structured information from scraped content - SIMPLIFIED"""
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
        HumanMessage(content="Extract the news articles and return only valid JSON array.")
    ]
    
    try:
        # Get LLM response
        print("🤖 Calling LLM for news extraction...")
        response = llm.invoke(messages)
        response_text = response.content.strip()
        
        # Simple JSON extraction - try multiple methods
        extracted_articles = None
        
        # Method 1: Try direct JSON parsing
        try:
            extracted_articles = json.loads(response_text)
        except:
            pass
        
        # Method 2: Remove code blocks and try again
        if not extracted_articles:
            try:
                clean_text = re.sub(r'```[a-z]*\n?', '', response_text)
                clean_text = re.sub(r'\n```', '', clean_text)
                extracted_articles = json.loads(clean_text.strip())
            except:
                pass
        
        # Method 3: Find JSON array in text
        if not extracted_articles:
            try:
                start = response_text.find('[')
                end = response_text.rfind(']') + 1
                if start >= 0 and end > start:
                    json_text = response_text[start:end]
                    extracted_articles = json.loads(json_text)
            except:
                pass
        
        # Ensure it's a list
        if extracted_articles and not isinstance(extracted_articles, list):
            extracted_articles = [extracted_articles]
        
        # If all methods failed, create fallback
        if not extracted_articles:
            print("⚠️ JSON parsing failed, creating fallback article")
            extracted_articles = [{
                "title": "Extraction Error - JSON Parse Failed",
                "summary": "Could not parse the article content properly",
                "publication_date": datetime.now(timezone.utc).date().isoformat(),
                "author": "TLDR",
                "article_url": "https://tldr.tech/",
                "image_url": "",
                "source": "TLDR",
                "main_topic": "Technology",
                "technologies": [],
                "content": response_text[:300] + "..." if len(response_text) > 300 else response_text
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
        
        # Return fallback data
        fallback_articles = [{
            "title": "System Error - Extraction Failed",
            "summary": f"Error occurred during extraction: {str(e)}",
            "publication_date": datetime.now(timezone.utc).date().isoformat(),
            "author": "TLDR",
            "article_url": "https://tldr.tech/",
            "image_url": "",
            "source": "TLDR",
            "main_topic": "Technology",
            "technologies": [],
            "content": "System error prevented content extraction"
        }]
        
        return {
            "extracted_articles": fallback_articles,
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Error in news extraction: {str(e)}")
            ]
        }
    
def analyze_relevance(state: AgentState) -> AgentState:
    """Node 3: This node analyzes the relevance of each article"""
    extracted_articles = state.get("extracted_articles", [])
    updated_articles = []
    print(f"🎯 RELEVANCE NODE: Got {len(extracted_articles)} articles to analyze")

    if not extracted_articles:
        return {
            "analyzed_articles": [],
            "messages": state.get("messages", []) + [
                SystemMessage(content="No extracted articles to analyze")
            ]
        }

    for i, article in enumerate(extracted_articles, 1):
        print(f"🔄 [{i}/{len(extracted_articles)}] Analyzing relevance...")
        
        try:
            system_prompt = news_relevance_score(
                article=article,
                user_expertise=["software development", "AI/ML", "Python development", "AI agent", "Javascript/React", "Next js", "Web development", "SaaS Apps"],
                target_audience=["Junior/Mid-level developers", "Startup Founders", "Freelance developers", "Tech leads"]
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content="Please analyze the relevance of this article and return ONLY valid JSON.")
            ]

            # Use safe LLM call with retries
            response = safe_llm_call(messages)
            
            # Use robust JSON parsing
            parsed = robust_json_parse(response.content)
            
            if not parsed:
                print(f"⚠️ Failed to parse JSON, skipping article {i}")
                continue

            if isinstance(parsed, dict):
                analysis = parsed
            elif isinstance(parsed, list) and parsed:
                analysis = parsed[0]
            else:
                print(f"⚠️ Unexpected JSON structure, skipping article {i}")
                continue

            article_copy = article.copy()
            article_copy.update({"relevance_analysis": analysis})
            updated_articles.append(article_copy)
            
            print(f"✅ Article {i} relevance analyzed successfully")
            
            # Small delay to avoid hitting rate limits
            time.sleep(0.5)

        except Exception as e:
            print(f"❌ Skipping article {i} due to error: {e}")
            continue

    print(f"🎉 Relevance analysis completed: {len(updated_articles)}/{len(extracted_articles)} articles")
    
    return {
        "analyzed_articles": updated_articles,
        "messages": state.get("messages", []) + [
            SystemMessage(content=f"Successfully analyzed {len(updated_articles)} articles for relevance")
        ]
    }

def categorize_articles(state: AgentState) -> AgentState:
    """Node 4: Categorize each article based on its content"""
    articles = state.get("analyzed_articles", [])
    updated_articles = []
    print(f"🏷️ CATEGORIZER NODE: Got {len(articles)} articles to categorize")

    if not articles:
        return {
            "analyzed_articles": [],
            "messages": state.get("messages", []) + [
                SystemMessage(content="No articles to categorize")
            ]
        }

    for i, article in enumerate(articles, 1):
        print(f"🔄 [{i}/{len(articles)}] Categorizing article...")
        
        try:
            system_prompt = article_categorizer(
                title=article["title"],
                content=article["content"],
                source=article["article_url"]
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content="Please categorize this article and return ONLY valid JSON.")
            ]

            # Use safe LLM call with retries
            response = safe_llm_call(messages)
            
            # Use robust JSON parsing
            parsed = robust_json_parse(response.content)
            
            if not parsed:
                print(f"⚠️ Failed to parse JSON, skipping article {i}")
                continue

            if isinstance(parsed, dict):
                category = parsed
            elif isinstance(parsed, list) and parsed:
                category = parsed[0]
            else:
                print(f"⚠️ Unexpected JSON structure, skipping article {i}")
                continue

            article_copy = article.copy()
            article_copy.update({"category": category})
            updated_articles.append(article_copy)
            
            print(f"✅ Article {i} categorized successfully")
            
            # Small delay to avoid hitting rate limits
            time.sleep(0.5)

        except Exception as e:
            print(f"❌ Skipping article {i} due to error: {e}")
            continue

    print(f"🎉 Categorization completed: {len(updated_articles)}/{len(articles)} articles")
    
    return {
        "analyzed_articles": updated_articles,
        "messages": state.get("messages", []) + [
            SystemMessage(content=f"Successfully categorized {len(updated_articles)} articles")
        ]
    }

def summarize_content(state: AgentState) -> AgentState:
    """Node 5: Summarize the content of each article"""
    articles = state.get("analyzed_articles", [])
    updated_articles = []
    print(f"📝 SUMMARIZER NODE: Got {len(articles)} articles to summarize")

    if not articles:
        return {
            "analyzed_articles": [],
            "messages": state.get("messages", []) + [
                SystemMessage(content="No articles to summarize")
            ]
        }

    for i, article in enumerate(articles, 1):
        print(f"🔄 [{i}/{len(articles)}] Summarizing article...")
        
        try:
            system_prompt = news_summarizer(
                article=article,
                source=article["article_url"],
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content="Summarize this article and return ONLY valid JSON.")
            ]

            # Use safe LLM call with retries
            response = safe_llm_call(messages)
            
            # Use robust JSON parsing
            parsed = robust_json_parse(response.content)
            
            if not parsed:
                print(f"⚠️ Failed to parse JSON, skipping article {i}")
                continue

            if isinstance(parsed, dict):
                analysis = parsed
            elif isinstance(parsed, list) and parsed:
                analysis = parsed[0]
            else:
                print(f"⚠️ Unexpected JSON structure, skipping article {i}")
                continue

            article_copy = article.copy()
            article_copy.update({"summary": analysis})
            updated_articles.append(article_copy)
            
            print(f"✅ Article {i} summarized successfully")
            
            # Small delay to avoid hitting rate limits
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ Skipping article {i} due to error: {e}")
            continue
    
    print(f"🎉 Summarization completed: {len(updated_articles)}/{len(articles)} articles")
    
    return {
        "analyzed_articles": updated_articles,
        "messages": state.get("messages", []) + [
            SystemMessage(content=f"Successfully summarized {len(updated_articles)} articles")
        ]
    }

def save_news_to_notion(state: AgentState) -> AgentState:
    """Node 6: Save analyzed articles to Notion database - FIXED VERSION"""
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
    
    def clean_multi_select(items):
        """Clean and split items safely for Notion multi_select"""
        clean = []
        if not items:
            return clean
            
        # Handle different input types
        if isinstance(items, str):
            items = [items]
        elif not isinstance(items, list):
            items = [str(items)]
            
        for item in items:
            if not item:
                continue
            # Split by commas if present
            parts = str(item).split(",")
            for part in parts:
                name = part.strip()
                if name:
                    clean.append(name[:100])  # Notion max length = 100
        return clean[:10]  # limit to 10 tags
    
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
                
                # Extract data from nested structures
                relevance_analysis = article.get("relevance_analysis", {})
                category_data = article.get("category", {})
                summary_data = article.get("summary", {})
                
                # Extract relevance score safely
                relevance_score = 5  # default
                if isinstance(relevance_analysis, dict):
                    relevance_score = relevance_analysis.get("relevance_score", 5)
                
                # Build properties safely - MATCHING YOUR ACTUAL DATABASE FIELDS
                properties = {
                    "Title": {
                        "title": [{"text": {"content": safe_text(article.get("title"), "Untitled")}}]
                    },
                    "URL": {
                        "url": article.get("article_url", "") if article.get("article_url", "").startswith("http") else ""
                    },
                    "Source": {
                        "rich_text": [{"text": {"content": safe_text(article.get("source"))}}]
                    },
                    "Author": {
                        "rich_text": [{"text": {"content": safe_text(article.get("author"))}}]
                    },
                    "Summary": {
                        "rich_text": [{"text": {"content": safe_text(article.get("content", ""), "")[:1900]}}]  # Use content as summary
                    },
                    "Content": {
                        "rich_text": [{"text": {"content": safe_text(article.get("content"), "")[:1900]}}]
                    },
                    "Main Topic": {
                        "rich_text": [{"text": {"content": safe_text(article.get("main_topic", "Technology"))}}]
                    },
                    "Status": {
                        "status": {"name": "Analyzed"}
                    },
                    "Post Created": {
                        "checkbox": False
                    },
                    "Scraped Date": {
                        "date": {"start": datetime.now().date().isoformat()}
                    },
                    "Relevance Score": {
                        "number": int(relevance_score)
                    }
                }
                
                # Handle publication date safely
                pub_date = article.get("publication_date")
                if pub_date and str(pub_date).strip():
                    try:
                        date_str = str(pub_date).strip()
                        if 'T' in date_str:
                            date_str = date_str.split('T')[0]
                        if len(date_str) >= 10:  # Basic validation YYYY-MM-DD
                            properties["Published Date"] = {
                                "date": {"start": date_str}
                            }
                    except Exception as date_error:
                        print(f"⚠️ Date parsing failed for {pub_date}: {date_error}")
                
                # Handle relevance analysis data safely
                if isinstance(relevance_analysis, dict):
                    # Content Category
                    content_category = relevance_analysis.get("content_category")
                    if content_category:
                        properties["Content Category"] = {
                            "rich_text": [{"text": {"content": safe_text(content_category)[:1900]}}]
                        }
                    
                    # Technical Depth
                    technical_depth = relevance_analysis.get("technical_depth")
                    if technical_depth:
                        properties["Technical Depth"] = {
                            "select": {"name": safe_text(technical_depth)[:100]}
                        }
                    
                    # Post Angle
                    post_angle = relevance_analysis.get("posting_angle")
                    if post_angle:
                        properties["Post Angle"] = {
                            "select": {"name": safe_text(post_angle)[:100]}
                        }
                    
                    # Audience Appeal
                    audience_appeal = relevance_analysis.get("audience_appeal")
                    if audience_appeal:
                        properties["Audience Appeal"] = {
                            "rich_text": [{"text": {"content": safe_text(audience_appeal)[:1900]}}]
                        }
                    
                    # Key Insights - CHANGED FROM "Key Insights" 
                    key_insights = relevance_analysis.get("key_insights", [])
                    if key_insights and isinstance(key_insights, list):
                        insights_text = "; ".join([str(insight) for insight in key_insights])
                        properties["Key Insights"] = {
                            "rich_text": [{"text": {"content": safe_text(insights_text)[:1900]}}]
                        }
                    
                    # Trending Potential
                    trending_data = relevance_analysis.get("trending_potential", {})
                    if isinstance(trending_data, dict):
                        trending_level = trending_data.get("level")
                        if trending_level:
                            properties["Engagement Potential"] = {
                                "select": {"name": safe_text(trending_level)[:100]}
                            }
                        
                        trending_reasoning = trending_data.get("reasoning")
                        if trending_reasoning:
                            properties["Trending Reasoning"] = {
                                "rich_text": [{"text": {"content": safe_text(trending_reasoning)[:1900]}}]
                            }
                
                # Handle category data
                if isinstance(category_data, dict):
                    primary_category = category_data.get("primary_category")
                    if primary_category:
                        properties["Primary Category"] = {
                            "select": {"name": safe_text(primary_category)[:100]}
                        }
                    
                    content_type = category_data.get("content_type")
                    if content_type:
                        properties["Content Type"] = {
                            "select": {"name": safe_text(content_type)[:100]}
                        }
                    
                    urgency_level = category_data.get("urgency_level")
                    if urgency_level:
                        properties["Urgency Level"] = {
                            "select": {"name": safe_text(urgency_level)[:100]}
                        }
                    
                    # Handle applicable categories
                    applicable_categories = category_data.get("applicable_categories", {})
                    if isinstance(applicable_categories, dict):
                        # Programming Languages
                        prog_langs = applicable_categories.get("programming_languages", [])
                        if prog_langs:
                            clean_langs = clean_multi_select(prog_langs)
                            if clean_langs:
                                properties["Programming Languages"] = {
                                    "multi_select": [{"name": lang} for lang in clean_langs]
                                }
                        
                        # Frameworks/Libraries - NOTE THE SPACE IN YOUR DB
                        frameworks = applicable_categories.get("frameworks_libraries", [])
                        if frameworks:
                            clean_frameworks = clean_multi_select(frameworks)
                            if clean_frameworks:
                                properties["Frameworks Libraries"] = {  # Note: no slash, just space
                                    "multi_select": [{"name": framework} for framework in clean_frameworks]
                                }
                        
                        # Industry Trends
                        trends = applicable_categories.get("industry_trends", [])
                        if trends:
                            clean_trends = clean_multi_select(trends)
                            if clean_trends:
                                properties["Industry Trends"] = {
                                    "multi_select": [{"name": trend} for trend in clean_trends]
                                }
                        
                        # Development Practices
                        practices = applicable_categories.get("development_practices", [])
                        if practices:
                            clean_practices = clean_multi_select(practices)
                            if clean_practices:
                                properties["Development Practices"] = {
                                    "multi_select": [{"name": practice} for practice in clean_practices]
                                }
                        
                        # Tools/Platforms - NOTE THE SPACE IN YOUR DB
                        tools = applicable_categories.get("tools_platforms", [])
                        if tools:
                            clean_tools = clean_multi_select(tools)
                            if clean_tools:
                                properties["Tools Platforms"] = {  # Note: no slash, just space
                                    "multi_select": [{"name": tool} for tool in clean_tools]
                                }
                
                # Handle summary data - REMOVE THE PROBLEMATIC FIELDS FOR NOW
                if isinstance(summary_data, dict):
                    # Technical Details
                    tech_details = summary_data.get("technical_details", [])
                    if tech_details:
                        clean_tech = clean_multi_select(tech_details)
                        if clean_tech:
                            properties["Technical Details"] = {
                                "multi_select": [{"name": tech} for tech in clean_tech]
                            }
                    
                    # Related Technologies
                    related_tech = summary_data.get("related_technologies", [])
                    if related_tech:
                        clean_related = clean_multi_select(related_tech)
                        if clean_related:
                            properties["Related Technologies"] = {
                                "multi_select": [{"name": tech} for tech in clean_related]
                            }
                
                # Handle Technologies from original article
                technologies = article.get("technologies", [])
                if technologies and isinstance(technologies, list):
                    clean_techs = clean_multi_select(technologies)
                    if clean_techs:
                        properties["Technologies"] = {
                            "multi_select": [{"name": tech} for tech in clean_techs]
                        }
                
                # Handle Image URL
                image_url = article.get("image_url")
                if image_url and isinstance(image_url, str) and image_url.startswith("http"):
                    properties["Image URL"] = {"url": image_url}
                
                # Create the database entry
                notion.pages.create(
                    parent={"database_id": database_id},
                    properties=properties
                )
                
                saved_count += 1
                print(f"✅ Saved: {article.get('title', 'Unknown')[:50]}...")
                
            except Exception as article_error:
                print(f"❌ Failed to save article '{article.get('title', 'Unknown')}': {article_error}")
                # Don't print full traceback for cleaner output, but you can uncomment this for debugging:
                # import traceback
                # traceback.print_exc()
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
        return {
            "analyzed_articles": articles,
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Error saving to Notion: {str(e)}")
            ]
        }

def save_repo_to_notion(state: AgentState) -> AgentState:
    """Node 7: Save analyzed repo activites to Notion database"""
    print("💾 STEP 6: Saving repository data to Notion database...")
    github_data = state.get("github_data", [])
    
    if not github_data:
        print("⚠️ No github data to save")
        return {
            "messages": state.get("messages", []) + [
                SystemMessage(content="No github data to save to Notion")
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
        database_id = os.getenv("DATABASE_ID") or os.getenv("GITHUB_DATABASE_ID")
        
        if not database_id:
            raise ValueError("Missing DATABASE_ID in environment variables")
            
        saved_count = 0
        print(f"📝 Attempting to save {len(github_data)} github datas...")
        
        for repo in github_data:
            try:
                print(f"🔄 Processing article: {repo.get('name', 'Unknown')}...")
                analysis = repo.get("analysis", {})
                
                # Build properties safely
                properties = {
                    "Repository Name": {
                        "title": [{"text": {"content": safe_text(repo.get("name"), "Untitled")}}]
                    },
                    "Repository URL": {
                        "url": repo.get("repo_url", "") or ""
                    },
                    "Description": {
                        "rich_text": [{"text": {"content": safe_text(repo.get("description"), "")}}]
                    },
                    "Content Summary": {
                        "rich_text": [{"text": {"content": safe_text(analysis.get("content_summary"), "")[:1900]}}]
                    },
                    "Key Hook": {
                        "rich_text": [{"text": {"content": safe_text(analysis.get("key_hook"), "")[:1900]}}]
                    },
                    "Diff Summary": {
                        "rich_text": [{"text": {"content": safe_text(analysis.get("diff_summary"), "")[:1900]}}]
                    },
                    "Primary Language": {
                        "rich_text": [{"text": {"content": safe_text(repo.get("language"), "")}}]
                    },
                    "Significance Level": {
                        "number": int(analysis.get("significance_level", "0"))  # safely default to 0
                    },
                    "Content Recommendation Verdict": {
                        "rich_text": [{"text": {"content": safe_text(analysis.get("content_recommendations", {}).get("verdict", ""), "")[:1900]}}]
                    },
                    "Story Problem-Solution": {
                        "rich_text": [{"text": {"content": safe_text(analysis.get("story_potential", {}).get("problem_solution", ""), "")[:1900]}}]
                    },
                    "Story Tech-Highlight": {
                        "rich_text": [{"text": {"content": safe_text(analysis.get("story_potential", {}).get("tech_highlight", ""), "")[:1900]}}]
                    },

                    "Developers Interest": {
                        "rich_text": [
                            {"text": {"content": safe_text(analysis.get("audience_interest", {}).get("developers", ""), "")[:1900]}}
                        ]
                    },
                    "Job Seekers Interest": {
                        "rich_text": [
                            {"text": {"content": safe_text(analysis.get("audience_interest", {}).get("job_seekers", ""), "")[:1900]}}
                        ]
                    },
                    "Product Managers Interest": {
                        "rich_text": [
                            {"text": {"content": safe_text(analysis.get("audience_interest", {}).get("product_managers", ""), "")[:1900]}}
                        ]
                    },
                    "Tech Leads Interest": {
                        "rich_text": [
                            {"text": {"content": safe_text(analysis.get("audience_interest", {}).get("tech_leads", ""), "")[:1900]}}
                        ]
                    },
                    "LinkedIn Post Created": {
                        "checkbox": False
                    },
                    "Status": {
                        "status": {"name": "Analyzed"}
                    },
                    "Posted Date": {
                        "date": {"start": datetime.now().date().isoformat()}
                    },
                    "Analysis Date": {
                        "date": {"start": datetime.now().date().isoformat()}
                    }
                }

                def clean_multi_select(items):
                    """Clean and split items safely for Notion multi_select"""
                    clean = []
                    for item in items:
                        if not item:
                            continue
                        # Split by commas if present
                        parts = str(item).split(",")
                        for part in parts:
                            name = part.strip()
                            if name:
                                clean.append(name[:100])  # Notion max length = 100
                    return clean[:10]  # limit to 10 tags


                
                technologies = analysis.get("technologies", [])
                if technologies and isinstance(technologies, list):
                    clean_techs = clean_multi_select(technologies)
                    if clean_techs:
                        properties["Tech Stack"] = {
                            "multi_select": [{"name": tech} for tech in clean_techs]
                        }
                professional_value = analysis.get("professional_value", [])
                if professional_value and isinstance(professional_value, list):
                    clean_techs = clean_multi_select(professional_value)
                    if clean_techs:
                        properties["Professional Value"] = {
                            "multi_select": [{"name": tech[:100]} for tech in clean_techs]
                        }
                technical_insights = analysis.get("technical_insights", [])
                if technical_insights and isinstance(technical_insights, list):
                    clean_techs = clean_multi_select(technical_insights)
                    if clean_techs:
                        properties["Technical Insights"] = {
                            "multi_select": [{"name": tech[:100]} for tech in clean_techs]
                        }
                hashtags = analysis.get("hashtags", [])
                if hashtags and isinstance(hashtags, list):
                    clean_techs = clean_multi_select(hashtags)
                    if clean_techs:
                        properties["Hashtags"] = {
                            "multi_select": [{"name": tech[:100]} for tech in clean_techs]
                        }
                content_angle = analysis.get("content_angles", [])
                if content_angle and isinstance(content_angle, list):
                    clean_techs = clean_multi_select(content_angle)
                    if clean_techs:
                        properties["Content Angle"] = {
                            "multi_select": [{"name": tech[:100]} for tech in clean_techs]
                        }
                top_formats = analysis.get("content_recommendations", {}).get("top_formats", [])
                if top_formats and isinstance(top_formats, list):
                    clean_techs = clean_multi_select(top_formats)
                    if clean_techs:
                        properties["Top Formats"] = {
                            "multi_select": [{"name": tech[:100]} for tech in clean_techs]
                        }
                # Create the database entry
                notion.pages.create(
                    parent={"database_id": database_id},
                    properties=properties
                )
                
                saved_count += 1
                print(f"✅ Saved: {repo.get('name', 'Unknown')[:50]}...")
                
            except Exception as repo_error:
                print(f"❌ Failed to save article '{repo.get('name', 'Unknown')}': {repo_error}")
                continue
        
        print(f"✅ Successfully saved {saved_count} of {len(github_data)} Repos data to Notion!")
        
        return {
            "github_data": github_data,
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Successfully saved {saved_count} of {len(github_data)} articles to Notion")
            ]
        }
        
    except Exception as e:
        print(f"❌ Notion saving failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "github_data": github_data,
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Error saving to Notion: {str(e)}")
            ]
        }


def post_content_strategist(state: AgentState) -> AgentState:
    """Node 8: This is the agent that generates strategic content recommendations"""
    print("🧠 STEP 8: Content Strategist - Analyzing content opportunities...")
    
    news_content = state.get("analyzed_articles", [])
    github_content = state.get("github_data", [])
    analyzed_content = []

    if not news_content and not github_content:
        print("⚠️ No content to analyze for strategy")
        return {
            "analyzed_content": [],
            "messages": state.get("messages", []) + [
                SystemMessage(content="No content available for strategic analysis")
            ]
        }

    # Process news articles for strategy
    print(f"📰 Strategically analyzing {len(news_content)} news articles...")
    for news in news_content:
        try:
            system_prompt = content_strategist(
                news_articles=news,
                repo_updates={},  # Empty for news-only analysis
                past_post_metrics=0,
                days_since_last_post=0,
                recent_topics="I've never posted",
                user_skills=["Python", "JavaScript / JSX", "TypeScript / TSX", "React.js", "Next.js", "Django",
                "FastAPI", "LangChain", "LangGraph", "LLM Prompt Engineering", "Playwright", "Browser-use Framework", 
                "Bright Data Proxy Integration","Firecrawl","PostgreSQL / MySQL","Notion API","REST API Design",
                "Authentication Systems (Django-based)","Matplotlib","Git & GitHub","uv / Virtual Environments",
                "Debugging & Logging","Docker Basics","Google Cloud AI Platform","Render"
                ],
                follower_demographics="My followers are mainly tech enthusiasts, developers, and AI/ML learners, with a mix of students, early professionals, and founders exploring the future of software and automation.",
                personal_brand="Building at the intersection of AI, automation, and software development — sharing insights, projects, and tools that make tech more practical and accessible."
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content="Generate strategic content recommendations for this news article."),
            ]

            response = llm.invoke(messages)
            clean_content = re.sub(r"```(?:json)?|```", "", response.content).strip()
            parsed = json.loads(clean_content)

            if isinstance(parsed, dict):
                strategy_analysis = parsed
            elif isinstance(parsed, list) and parsed:
                strategy_analysis = parsed[0]
            else:
                raise ValueError(f"Unexpected JSON structure: {parsed}")

            # Create strategic content item
            strategic_item = news.copy()
            strategic_item.update({
                "content_type": "news_article",
                "strategy_analysis": strategy_analysis,
                "source_material": {
                    "title": news.get("title", ""),
                    "content": news.get("content", ""),
                    "summary": news.get("summary", {}),
                    "relevance_analysis": news.get("relevance_analysis", {})
                }
            })
            analyzed_content.append(strategic_item)
            
            print(f"✅ Strategic analysis completed for: {news.get('title', 'Unknown')[:50]}...")

        except Exception as e:
            print(f"❌ Content Strategist failed for news article: {str(e)}")
            continue

    # Process GitHub repositories for strategy
    print(f"🔄 Strategically analyzing {len(github_content)} GitHub repositories...")
    for github in github_content:
        try:
            system_prompt = content_strategist(
                news_articles={},  # Empty for repo-only analysis
                repo_updates=github,
                past_post_metrics=0,
                days_since_last_post=0,
                recent_topics="I've never posted",
                user_skills=["Python", "JavaScript / JSX", "TypeScript / TSX", "React.js", "Next.js", "Django",
                "FastAPI", "LangChain", "LangGraph", "LLM Prompt Engineering", "Playwright", "Browser-use Framework", 
                "Bright Data Proxy Integration","Firecrawl","PostgreSQL / MySQL","Notion API","REST API Design",
                "Authentication Systems (Django-based)","Matplotlib","Git & GitHub","uv / Virtual Environments",
                "Debugging & Logging","Docker Basics","Google Cloud AI Platform","Render"
                ],
                follower_demographics="My followers are mainly tech enthusiasts, developers, and AI/ML learners, with a mix of students, early professionals, and founders exploring the future of software and automation.",
                personal_brand="Building at the intersection of AI, automation, and software development — sharing insights, projects, and tools that make tech more practical and accessible."
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content="Generate strategic content recommendations for this GitHub repository."),
            ]

            response = llm.invoke(messages)
            clean_content = re.sub(r"```(?:json)?|```", "", response.content).strip()
            parsed = json.loads(clean_content)

            if isinstance(parsed, dict):
                strategy_analysis = parsed
            elif isinstance(parsed, list) and parsed:
                strategy_analysis = parsed[0]
            else:
                raise ValueError(f"Unexpected JSON structure: {parsed}")

            # Create strategic content item
            strategic_item = github.copy()
            strategic_item.update({
                "content_type": "github_project",
                "strategy_analysis": strategy_analysis,
                "source_material": {
                    "name": github.get("name", ""),
                    "description": github.get("description", ""),
                    "analysis": github.get("analysis", {}),
                    "repo_url": github.get("repo_url", "")
                }
            })
            analyzed_content.append(strategic_item)
            
            print(f"✅ Strategic analysis completed for repo: {github.get('name', 'Unknown')}")

        except Exception as e:
            print(f"❌ Content Strategist failed for GitHub repo: {str(e)}")
            continue

    print(f"🎯 Content Strategist completed! Analyzed {len(analyzed_content)} content opportunities")

    return {
        "analyzed_content": analyzed_content,
        "messages": state.get("messages", []) + [
            SystemMessage(content=f"Successfully analyzed {len(analyzed_content)} content opportunities")
        ]
    }


def content_writer_agent(state: AgentState) -> AgentState:
    """Agent that generates LinkedIn posts from strategic content analysis"""
    print("✍️ STEP 9: Content Writer Agent - Generating LinkedIn posts...")
    
    strategy_content = state.get("analyzed_content", [])

    if not strategy_content:
        print("⚠️ No strategic content available for Content Writer Agent")
        return {
            "linkedin_posts": [],
            "messages": state.get("messages", []) + [
                SystemMessage(content="No strategic content available to generate LinkedIn posts")
            ]
        }

    linkedin_posts = []
    print(f"📝 Processing {len(strategy_content)} strategic content items...")

    for item in strategy_content:
        try:
            content_type = item.get("content_type", "unknown")
            strategy_analysis = item.get("strategy_analysis", {})
            source_material = item.get("source_material", {})
            
            # Extract strategic insights for the post
            if isinstance(strategy_analysis, dict):
                recommended_content_type = strategy_analysis.get("recommended_content_type", "Commentary")
                content_angle = strategy_analysis.get("content_angle", "Professional insight")
                primary_topic_focus = strategy_analysis.get("primary_topic_focus", "Technology")
                reasoning = strategy_analysis.get("reasoning", "")
            else:
                recommended_content_type = "Commentary"
                content_angle = "Professional insight" 
                primary_topic_focus = "Technology"
                reasoning = ""

            # Determine source material text based on content type
            if content_type == "news_article":
                source_text = f"Title: {source_material.get('title', '')}\nContent: {source_material.get('content', '')[:500]}..."
                # Extract key insights from news analysis
                relevance_data = source_material.get('relevance_analysis', {})
                summary_data = source_material.get('summary', {})
                key_insights = []
                
                if isinstance(relevance_data, dict):
                    key_insights.extend(relevance_data.get('key_insights', []))
                if isinstance(summary_data, dict):
                    key_insights.extend(summary_data.get('key_points', []))
                    
            elif content_type == "github_project":
                repo_analysis = source_material.get('analysis', {})
                source_text = f"Project: {source_material.get('name', '')}\nDescription: {source_material.get('description', '')}"
                if isinstance(repo_analysis, dict):
                    content_summary = repo_analysis.get('content_summary', '')
                    if content_summary:
                        source_text += f"\nSummary: {content_summary}"
                    key_insights = repo_analysis.get('technical_insights', [])
                else:
                    key_insights = []
            else:
                source_text = str(source_material)
                key_insights = []

            # 1. Generate base LinkedIn post
            print(f"🎯 Generating post for {content_type}: {source_material.get('title') or source_material.get('name', 'Unknown')[:30]}...")
            
            system_prompt_writer = linkedin_post_writer(
                content_type=recommended_content_type,
                source_material=source_text,
                previous_posts="Professional but conversational software developer voice with authentic personal insights",
                audience="Developers, founders, and tech enthusiasts",
                content_angle=content_angle,
                key_insights=key_insights[:3] if key_insights else [primary_topic_focus]
            )
            
            messages_writer = [
                SystemMessage(content=system_prompt_writer),
                HumanMessage(content=f"Write a LinkedIn post based on the strategic analysis. Focus on: {reasoning[:100]}...")
            ]
            
            writer_response = llm.invoke(messages_writer)
            clean_writer_response = re.sub(r"```(?:json)?|```", "", writer_response.content).strip()
            base_post_data = json.loads(clean_writer_response)

            base_post_text = base_post_data.get("post_content", "")

            # 2. Generate variations of the post
            print(f"🔄 Generating variations...")
            system_prompt_variations = post_variation_generator(
                original_concept=base_post_text,
                audience="Developers, founders, and tech enthusiasts",
                voice_sample="Professional but conversational software developer voice with authentic personal insights"
            )
            
            messages_variations = [
                SystemMessage(content=system_prompt_variations),
                HumanMessage(content="Generate 3 variations of the LinkedIn post that maintain the strategic focus.")
            ]
            
            variation_response = llm.invoke(messages_variations)
            clean_variation_response = re.sub(r"```(?:json)?|```", "", variation_response.content).strip()
            variations_data = json.loads(clean_variation_response)

            # Store the complete result
            linkedin_posts.append({
                "content_type": content_type,
                "source_title": source_material.get('title') or source_material.get('name', 'Unknown'),
                "strategic_reasoning": reasoning,
                "recommended_content_type": recommended_content_type,
                "content_angle": content_angle,
                "base_post": base_post_data,
                "variations": variations_data,
                "strategy_analysis": strategy_analysis,
                "source_data": source_material
            })
            
            print(f"✅ Generated LinkedIn post with variations")

        except Exception as e:
            print(f"❌ Content Writer Agent failed for item: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"🎉 Content Writer Agent completed! Generated {len(linkedin_posts)} LinkedIn posts")

    return {
        "linkedin_posts": linkedin_posts,
        "messages": state.get("messages", []) + [
            SystemMessage(content=f"Successfully generated {len(linkedin_posts)} LinkedIn posts with strategic variations")
        ]
    }

# def save_linkedin_posts_to_notion(state: AgentState) -> AgentState:
#     """Node: Save generated LinkedIn posts to Notion database - FIXED VERSION"""
#     print("💾 STEP 10: Saving LinkedIn posts to Notion database...")
#     linkedin_posts = state.get("linkedin_posts", [])
    
#     if not linkedin_posts:
#         print("⚠️ No LinkedIn posts to save")
#         return {
#             "messages": state.get("messages", []) + [
#                 SystemMessage(content="No LinkedIn posts to save to Notion")
#             ]
#         }
    
#     # Helper function to safely get text values
#     def safe_text(text_value, default="", max_length=2000):
#         """Safely convert value to string, handling None/null values"""
#         if text_value is None:
#             return default
#         text_str = str(text_value).strip()
#         if not text_str:
#             return default
#         # Truncate if too long for Notion
#         return text_str[:max_length] if len(text_str) > max_length else text_str
    
#     def safe_select_text(text_value, default="General", max_length=100):
#         """Safely convert text for Notion select field (no commas, limited length)"""
#         if text_value is None:
#             return default
        
#         text_str = str(text_value).strip()
#         if not text_str:
#             return default
            
#         # Remove commas and other problematic characters for select fields
#         clean_text = text_str.replace(",", " -").replace("\n", " ").replace("\r", " ")
        
#         # Truncate and clean up multiple spaces
#         clean_text = " ".join(clean_text.split())
        
#         return clean_text[:max_length] if len(clean_text) > max_length else clean_text
    
#     def clean_multi_select(items, max_items=10):
#         """Clean and prepare items for Notion multi_select"""
#         if not items or not isinstance(items, list):
#             return []
        
#         clean_items = []
#         for item in items[:max_items]:  # Limit to max_items
#             if item and str(item).strip():
#                 # Clean the item and remove commas for multi-select
#                 clean_item = str(item).strip().replace(",", " -")[:100]
#                 if clean_item:
#                     clean_items.append({"name": clean_item})
#         return clean_items
    
#     def extract_hashtags(post_data):
#         """Extract hashtags from post data"""
#         if not isinstance(post_data, dict):
#             return []
#         return post_data.get("hashtags", [])
    
#     try:
#         # Initialize Notion client
#         notion = Client(auth=os.getenv("NOTION_TOKEN"))
#         database_id = os.getenv("LINKEDIN_POSTS_DATABASE_ID")
        
#         if not database_id:
#             raise ValueError("Missing LINKEDIN_POSTS_DATABASE_ID in environment variables")
            
#         saved_count = 0
#         print(f"📝 Attempting to save {len(linkedin_posts)} LinkedIn posts...")
        
#         for post in linkedin_posts:
#             try:
#                 # Extract main data
#                 source_title = safe_text(post.get("source_title", "Unknown"))
#                 content_type = safe_select_text(post.get("content_type", "unknown"))
#                 strategic_reasoning = safe_text(post.get("strategic_reasoning", ""))
#                 content_angle = safe_select_text(post.get("content_angle", "General"))
#                 recommended_content_type = safe_select_text(post.get("recommended_content_type", "Commentary"))
                
#                 # Extract base post data
#                 base_post = post.get("base_post", {})
#                 post_content = safe_text(base_post.get("post_content", ""), max_length=1900)
                
#                 # Extract post structure for additional info
#                 post_structure = base_post.get("post_structure", {})
#                 hook = safe_text(post_structure.get("hook", ""))
#                 call_to_action = safe_text(post_structure.get("call_to_action", ""))
                
#                 # Extract variations
#                 variations = post.get("variations", {})
#                 variation_a = ""
#                 variation_b = ""
#                 variation_c = ""
                
#                 if isinstance(variations, dict):
#                     variations_data = variations.get("variations", {})
#                     if isinstance(variations_data, dict):
#                         variation_a = safe_text(variations_data.get("version_a_news_commentary", {}).get("content", ""), max_length=1900)
#                         variation_b = safe_text(variations_data.get("version_b_personal_experience", {}).get("content", ""), max_length=1900)
#                         variation_c = safe_text(variations_data.get("version_c_community_discussion", {}).get("content", ""), max_length=1900)
                
#                 # Extract hashtags
#                 hashtags = extract_hashtags(base_post)
                
#                 # Extract source data
#                 source_data = post.get("source_data", {})
#                 source_url = ""
#                 if isinstance(source_data, dict):
#                     source_url = source_data.get("article_url") or source_data.get("repo_url", "")
                
#                 # Determine engagement potential based on strategy analysis
#                 strategy_analysis = post.get("strategy_analysis", {})
#                 engagement_potential = "Medium"  # default
#                 content_quality_score = 5  # default
                
#                 if isinstance(strategy_analysis, dict):
#                     content_score = strategy_analysis.get("content_opportunity_score", 5)
#                     if isinstance(content_score, (int, float)):
#                         content_quality_score = int(content_score)
#                         if content_score >= 8:
#                             engagement_potential = "High"
#                         elif content_score <= 4:
#                             engagement_potential = "Low"
                
#                 # Build properties for Notion
#                 properties = {
#                     "Post Title": {
#                         "title": [{"text": {"content": source_title[:100]}}]  # Title field limit
#                     },
#                     "Post Content": {
#                         "rich_text": [{"text": {"content": post_content}}]
#                     },
#                     "Content Type": {
#                         "select": {"name": content_type}
#                     },
#                     "Source Title": {
#                         "rich_text": [{"text": {"content": source_title}}]
#                     },
#                     "Strategic Reasoning": {
#                         "rich_text": [{"text": {"content": strategic_reasoning}}]
#                     },
#                     "Content Angle": {
#                         "select": {"name": content_angle}
#                     },
#                     "Recommended Content Type": {
#                         "select": {"name": recommended_content_type}
#                     },
#                     "Post Status": {
#                         "status": {"name": "Generated"}
#                     },
#                     "Generated Date": {
#                         "date": {"start": datetime.now().date().isoformat()}
#                     },
#                     "Posting Priority": {
#                         "select": {"name": "Medium"}
#                     },
#                     "Engagement Potential": {
#                         "select": {"name": engagement_potential}
#                     },
#                     "Content Quality Score": {
#                         "number": content_quality_score
#                     },
#                     "Technical Relevance": {
#                         "select": {"name": "High" if content_type in ["github_project", "tutorial"] else "Medium"}
#                     },
#                     "Hook Strength": {
#                         "number": 7  # Default, could be analyzed by LLM later
#                     }
#                 }
                
#                 # Add optional fields
#                 if source_url and source_url.startswith("http"):
#                     properties["Source URL"] = {"url": source_url}
                
#                 if call_to_action:
#                     properties["Call to Action"] = {
#                         "rich_text": [{"text": {"content": call_to_action}}]
#                     }
                
#                 # Add hashtags if available
#                 if hashtags:
#                     clean_hashtags = clean_multi_select(hashtags)
#                     if clean_hashtags:
#                         properties["Hashtags"] = {"multi_select": clean_hashtags}
                
#                 # Add variations if they exist
#                 if variation_a:
#                     properties["Variation A - News Commentary"] = {
#                         "rich_text": [{"text": {"content": variation_a}}]
#                     }
                
#                 if variation_b:
#                     properties["Variation B - Personal Experience"] = {
#                         "rich_text": [{"text": {"content": variation_b}}]
#                     }
                
#                 if variation_c:
#                     properties["Variation C - Community Discussion"] = {
#                         "rich_text": [{"text": {"content": variation_c}}]
#                     }
                
#                 # Set target audience based on content type
#                 audience_options = []
#                 if content_type == "github_project":
#                     audience_options = [{"name": "Developers"}, {"name": "Tech Enthusiasts"}]
#                 elif content_type == "news_article":
#                     audience_options = [{"name": "Developers"}, {"name": "Founders"}, {"name": "Tech Enthusiasts"}]
#                 else:
#                     audience_options = [{"name": "Tech Enthusiasts"}]
                
#                 if audience_options:
#                     properties["Target Audience"] = {"multi_select": audience_options}
                
#                 # Create the Notion page
#                 notion.pages.create(
#                     parent={"database_id": database_id},
#                     properties=properties
#                 )
                
#                 saved_count += 1
#                 print(f"✅ Saved LinkedIn post: {source_title[:50]}...")
                
#             except Exception as post_error:
#                 print(f"❌ Failed to save LinkedIn post '{post.get('source_title', 'Unknown')}': {post_error}")
#                 # Uncomment for debugging:
#                 # import traceback
#                 # traceback.print_exc()
#                 continue
        
#         print(f"🎉 Successfully saved {saved_count} of {len(linkedin_posts)} LinkedIn posts to Notion!")
        
#         return {
#             "linkedin_posts": linkedin_posts,
#             "messages": state.get("messages", []) + [
#                 SystemMessage(content=f"Successfully saved {saved_count} of {len(linkedin_posts)} LinkedIn posts to Notion")
#             ]
#         }
        
#     except Exception as e:
#         print(f"❌ Notion saving failed: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return {
#             "linkedin_posts": linkedin_posts,
#             "messages": state.get("messages", []) + [
#                 SystemMessage(content=f"Error saving LinkedIn posts to Notion: {str(e)}")
#             ]
#         }

def content_reviewer_agent(state: AgentState) -> AgentState:
    """Node 13: Reviews generated posts for quality, brand alignment, and professional standards"""
    linkedin_posts = state.get("linkedin_posts", [])
    reviewed_posts = []

    if not linkedin_posts:
        print("No linkein posts available for content reviewer agent")
        return {
            "linkedin_posts": [],
            "messages": state.get("messages", []) + [
                SystemMessage(content="No posts available for LinkedIn posts review")
            ]
        }

    print(f"📋 Reviewing {len(linkedin_posts)} generated posts...")

    for i, post in enumerate(linkedin_posts, 1):
        try:
            
            # Extract the main post content for review
            base_post = post.get("base_post", {})
            post_content = base_post.get("post_content", "")
            content_type = post.get("content_type", "unknown")
            
            if not post_content:
                print(f"⚠️ Skipping post {i} - no content to review")
                continue
            
            # Generate review using your existing prompt
            system_prompt = content_reviewer(
                post_draft=post_content,
                brand_voice="Professional but conversational software developer sharing authentic insights and practical knowledge",
                audience="Developers, founders, and tech enthusiasts interested in AI, automation, and modern software development",
                topic=content_type
            )
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content="Review this LinkedIn post for final quality control and brand alignment.")
            ]
            
            response = safe_llm_call(messages)
            review_data = robust_json_parse(response.content)
            
            if not review_data:
                print(f"⚠️ Failed to parse review for post {i}, using default approval")
                review_data = {
                    "approval_status": "Approve",
                    "reasoning": "Default approval due to parsing error",
                    "quality_assessment": {"brand_alignment": {"score": 7}}
                }
            
            # Extract review decision
            approval_status = review_data.get("approval_status", "Approve")
            reasoning = review_data.get("reasoning", "Standard review")
            quality_score = review_data.get("quality_assessment", {}).get("brand_alignment", {}).get("score", 7)
            
            # Add review data to the post
            reviewed_post = post.copy()
            reviewed_post.update({
                "review_result": review_data,
                "approval_status": approval_status,
                "review_reasoning": reasoning,
                "final_quality_score": quality_score,
                "reviewed_at": datetime.now().isoformat()
            })
            
            reviewed_posts.append(reviewed_post)
            
            print(f"✅ Review completed: {approval_status} (Quality: {quality_score}/10)")
            
            # Small delay to avoid rate limits
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ Review failed for post {i}: {e}")
            # Add post with default approval on error
            error_post = post.copy()
            error_post.update({
                "review_result": {"approval_status": "Approve", "reasoning": f"Review error: {str(e)}"},
                "approval_status": "Approve",
                "review_reasoning": f"Default approval due to review error: {str(e)}",
                "final_quality_score": 6,
                "reviewed_at": datetime.now().isoformat()
            })
            reviewed_posts.append(error_post)
            continue
    
    approved_count = len([p for p in reviewed_posts if p.get("approval_status") == "Approve"])
    print(f"🎯 Content Review Summary: {approved_count}/{len(reviewed_posts)} posts approved")
    
    return {
        "reviewed_posts": reviewed_posts,
        "messages": state.get("messages", []) + [
            SystemMessage(content=f"Content review completed: {approved_count}/{len(reviewed_posts)} posts approved")
        ]
    }

def scheduling_optimizer_agent(state: AgentState) -> AgentState:
    """NEW Node: Determine optimal posting times for all approved posts"""
    print("⏰ STEP 12: Scheduling Optimizer - Planning optimal posting times...")
    
    reviewed_posts = state.get("reviewed_posts", [])
    
    if not reviewed_posts:
        print("⚠️ No reviewed posts to schedule")
        return {
            "scheduled_posts": [],
            "messages": state.get("messages", []) + [
                SystemMessage(content="No posts available for scheduling")
            ]
        }
    
    # Filter only approved posts
    approved_posts = [p for p in reviewed_posts if p.get("approval_status") == "Approve"]
    print(f"📅 Scheduling {len(approved_posts)} approved posts...")
    
    scheduled_posts = []
    base_date = datetime.now().date()
    
    for i, post in enumerate(approved_posts):
        try:
            
            content_type = post.get("content_type", "General")
            source_title = post.get("source_title", "Tech Content")
            final_quality_score = post.get("final_quality_score", 7)
            
            # Generate optimal timing using your existing prompt
            system_prompt = posting_timing_optimizer(
                content_type=content_type,
                topic=source_title,
                audience_segments=["Software Developers", "Tech Founders", "AI/ML Engineers", "Startup Enthusiasts"],
                timezone="UTC",
                current_time=datetime.now().isoformat(),
                recent_post_times=[]  # TODO: Could fetch from historical data
            )
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Determine optimal posting schedule for this {content_type} content about {source_title}.")
            ]
            
            response = safe_llm_call(messages)
            timing_data = robust_json_parse(response.content)
            
            if not timing_data:
                print(f"⚠️ Failed to parse timing for post {i+1}, using default schedule")
                timing_data = {
                    "optimal_posting_time": {
                        "recommended_datetime": f"{base_date + timedelta(days=i)} 14:00:00 UTC",
                        "reasoning": "Default scheduling due to parsing error"
                    }
                }
            
            # Extract timing recommendations
            optimal_time = timing_data.get("optimal_posting_time", {})
            recommended_datetime = optimal_time.get("recommended_datetime", f"{base_date + timedelta(days=i)} 14:00:00 UTC")
            timing_reasoning = optimal_time.get("reasoning", "Standard professional hours")
            
            # Parse and normalize the recommended datetime
            try:
                # Handle various datetime formats from LLM
                if "Today" in recommended_datetime or "today" in recommended_datetime:
                    scheduled_datetime = datetime.combine(base_date, datetime.strptime("14:00", "%H:%M").time())
                elif "Tomorrow" in recommended_datetime or "tomorrow" in recommended_datetime:
                    scheduled_datetime = datetime.combine(base_date + timedelta(days=1), datetime.strptime("14:00", "%H:%M").time())
                else:
                    # Try to parse the datetime string
                    try:
                        # Handle various formats
                        for fmt in ["%Y-%m-%d %H:%M:%S UTC", "%Y-%m-%d %H:%M UTC", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]:
                            try:
                                scheduled_datetime = datetime.strptime(recommended_datetime.replace(" UTC", ""), fmt.replace(" UTC", ""))
                                break
                            except ValueError:
                                continue
                        else:
                            # Default if parsing fails
                            scheduled_datetime = datetime.combine(base_date + timedelta(days=i), datetime.strptime("14:00", "%H:%M").time())
                    except:
                        scheduled_datetime = datetime.combine(base_date + timedelta(days=i), datetime.strptime("14:00", "%H:%M").time())
            except:
                # Ultimate fallback
                scheduled_datetime = datetime.combine(base_date + timedelta(days=i), datetime.strptime("14:00", "%H:%M").time())
            
            # Create scheduled post
            scheduled_post = post.copy()
            scheduled_post.update({
                "scheduled_datetime": scheduled_datetime.isoformat(),
                "scheduling_data": timing_data,
                "timing_reasoning": timing_reasoning,
                "posting_priority": "High" if final_quality_score >= 8 else "Medium" if final_quality_score >= 6 else "Low",
                "scheduled_at": datetime.now().isoformat(),
                "ready_for_publishing": True
            })
            
            scheduled_posts.append(scheduled_post)
            
            print(f"✅ Scheduled for: {scheduled_datetime.strftime('%Y-%m-%d %H:%M UTC')} (Priority: {scheduled_post['posting_priority']})")
            
            # Small delay
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ Scheduling failed for post {i+1}: {e}")
            # Add with default scheduling
            default_scheduled_post = post.copy()
            default_scheduled_post.update({
                "scheduled_datetime": datetime.combine(base_date + timedelta(days=i), datetime.strptime("14:00", "%H:%M").time()).isoformat(),
                "scheduling_data": {"error": str(e)},
                "timing_reasoning": f"Default scheduling due to error: {str(e)}",
                "posting_priority": "Medium",
                "scheduled_at": datetime.now().isoformat(),
                "ready_for_publishing": True
            })
            scheduled_posts.append(default_scheduled_post)
            continue
    
    print(f"⏰ Scheduling Summary: {len(scheduled_posts)} posts scheduled for optimal times")
    
    return {
        "scheduled_posts": scheduled_posts,
        "messages": state.get("messages", []) + [
            SystemMessage(content=f"Scheduling optimization completed: {len(scheduled_posts)} posts ready for publishing")
        ]
    }

def save_scheduled_posts_to_notion(state: AgentState) -> AgentState:
    """NEW Node: Save scheduled posts to Notion with timing and priority data"""
    print("💾 STEP 13: Saving scheduled posts to Notion...")
    
    scheduled_posts = state.get("scheduled_posts", [])
    
    if not scheduled_posts:
        print("⚠️ No scheduled posts to save")
        return {
            "messages": state.get("messages", []) + [
                SystemMessage(content="No scheduled posts to save")
            ]
        }
    
    def safe_text(text_value, default="", max_length=2000):
        if text_value is None:
            return default
        text_str = str(text_value).strip()
        if not text_str:
            return default
        return text_str[:max_length] if len(text_str) > max_length else text_str
    
    def safe_select_text(text_value, default="Medium", max_length=100):
        if text_value is None:
            return default
        text_str = str(text_value).strip()
        if not text_str:
            return default
        clean_text = text_str.replace(",", " -").replace("\n", " ").replace("\r", " ")
        clean_text = " ".join(clean_text.split())
        return clean_text[:max_length] if len(clean_text) > max_length else clean_text
    
    try:
        notion = Client(auth=os.getenv("NOTION_TOKEN"))
        database_id = os.getenv("LINKEDIN_POSTS_DATABASE_ID")
        
        if not database_id:
            raise ValueError("Missing LINKEDIN_POSTS_DATABASE_ID")
        
        saved_count = 0
        
        for post in scheduled_posts:
            try:
                # Extract all the data
                source_title = safe_text(post.get("source_title", "Unknown"))
                base_post = post.get("base_post", {})
                post_content = safe_text(base_post.get("post_content", ""), max_length=1900)
                
                # Scheduling data
                scheduled_datetime = post.get("scheduled_datetime", "")
                posting_priority = safe_select_text(post.get("posting_priority", "Medium"))
                timing_reasoning = safe_text(post.get("timing_reasoning", ""))
                
                # Review data
                approval_status = post.get("approval_status", "Approve")
                final_quality_score = post.get("final_quality_score", 7)
                review_reasoning = safe_text(post.get("review_reasoning", ""))
                
                # Parse scheduled datetime for Notion
                scheduled_date = ""
                scheduled_time = ""
                try:
                    dt = datetime.fromisoformat(scheduled_datetime.replace('Z', '+00:00'))
                    scheduled_date = dt.date().isoformat()
                    scheduled_time = dt.time().strftime("%H:%M")
                except:
                    scheduled_date = datetime.now().date().isoformat()
                    scheduled_time = "14:00"
                
                # Build properties for Notion
                properties = {
                    "Post Title": {
                        "title": [{"text": {"content": source_title[:100]}}]
                    },
                    "Post Content": {
                        "rich_text": [{"text": {"content": post_content}}]
                    },
                    "Post Status": {
                        "status": {"name": "Scheduled"}  # NEW STATUS
                    },
                    "Scheduled Date": {
                        "date": {"start": scheduled_date}
                    },
                    "Scheduled Time": {
                        "rich_text": [{"text": {"content": scheduled_time}}]
                    },
                    "Posting Priority": {
                        "select": {"name": posting_priority}
                    },
                    "Content Quality Score": {
                        "number": int(final_quality_score)
                    },
                    "Approval Status": {
                        "select": {"name": approval_status}
                    },
                    "Review Reasoning": {
                        "rich_text": [{"text": {"content": review_reasoning}}]
                    },
                    "Timing Reasoning": {
                        "rich_text": [{"text": {"content": timing_reasoning}}]
                    },
                    "Ready for Publishing": {
                        "checkbox": True
                    },
                    "Generated Date": {
                        "date": {"start": datetime.now().date().isoformat()}
                    },
                    "LinkedIn Post Created": {
                        "checkbox": False  # Will be set to True after actual publishing
                    }
                }
                
                # Add other existing fields
                content_type = safe_select_text(post.get("content_type", "General"))
                properties["Content Type"] = {"select": {"name": content_type}}
                
                engagement_potential = safe_select_text(post.get("engagement_potential", "Medium"))
                properties["Engagement Potential"] = {"select": {"name": engagement_potential}}
                
                # Add variations if they exist
                variations = post.get("variations", {})
                if isinstance(variations, dict):
                    variations_data = variations.get("variations", {})
                    if isinstance(variations_data, dict):
                        variation_a = safe_text(variations_data.get("version_a_news_commentary", {}).get("content", ""), max_length=1900)
                        variation_b = safe_text(variations_data.get("version_b_personal_experience", {}).get("content", ""), max_length=1900)
                        variation_c = safe_text(variations_data.get("version_c_community_discussion", {}).get("content", ""), max_length=1900)
                        
                        if variation_a:
                            properties["Variation A - News Commentary"] = {"rich_text": [{"text": {"content": variation_a}}]}
                        if variation_b:
                            properties["Variation B - Personal Experience"] = {"rich_text": [{"text": {"content": variation_b}}]}
                        if variation_c:
                            properties["Variation C - Community Discussion"] = {"rich_text": [{"text": {"content": variation_c}}]}
                
                # Create the Notion page
                notion.pages.create(
                    parent={"database_id": database_id},
                    properties=properties
                )
                
                saved_count += 1
                print(f"✅ Saved scheduled post: {source_title[:50]}... (Priority: {posting_priority}, Date: {scheduled_date} {scheduled_time})")
                
            except Exception as post_error:
                print(f"❌ Failed to save scheduled post: {post_error}")
                continue
        
        print(f"🎉 Successfully saved {saved_count}/{len(scheduled_posts)} scheduled posts to Notion!")
        
        return {
            "scheduled_posts": scheduled_posts,
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Successfully saved {saved_count}/{len(scheduled_posts)} scheduled posts")
            ]
        }
        
    except Exception as e:
        print(f"❌ Failed to save scheduled posts: {e}")
        return {
            "scheduled_posts": scheduled_posts,
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Error saving scheduled posts: {str(e)}")
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
graph.add_node("notion_news_saver", save_news_to_notion)
graph.add_node("github_analyzer", analyze_github_repos)
graph.add_node("notion_github_saver", save_repo_to_notion)
graph.add_node("content_strategist", post_content_strategist)
graph.add_node("content_writer", content_writer_agent)
graph.add_node("content_reviewer", content_reviewer_agent) 
graph.add_node("scheduling_optimizer", scheduling_optimizer_agent) 
graph.add_node("scheduled_posts_saver", save_scheduled_posts_to_notion) 

# Define the flow - FIXED VERSION
graph.add_edge(START, "scraper")
graph.add_edge("scraper", "extractor") 
graph.add_edge("extractor", "relevance_analyst")
graph.add_edge("relevance_analyst", "categorizer")
graph.add_edge("categorizer", "summarizer")
graph.add_edge("summarizer", "notion_news_saver")
graph.add_edge("notion_news_saver", "github_analyzer")
graph.add_edge("github_analyzer", "notion_github_saver")
graph.add_edge("notion_github_saver", "content_strategist")
graph.add_edge("content_strategist", "content_writer")
graph.add_edge("content_writer", "content_reviewer")  # NEW EDGE
graph.add_edge("content_reviewer", "scheduling_optimizer")  # NEW EDGE
graph.add_edge("scheduling_optimizer", "scheduled_posts_saver")  # NEW EDGE
graph.add_edge("scheduled_posts_saver", END)   


# Compile the graph
app = graph.compile()

# Enhanced run function with better progress indicators

def run_weekly_content_generation():
    """Enhanced weekly content generation with review and scheduling"""
    print("🚀 ENHANCED WEEKLY CONTENT PIPELINE")
    print("🎯 Content Generation → Strategy → Review → Scheduling")
    print("=" * 60)
    
    initial_state = {
        "messages": [],
        "raw_content": [],
        "extracted_articles": [],
        "analyzed_articles": [],
        "github_data": [],
        "analyzed_content": [],
        "linkedin_posts": [],
        "reviewed_posts": [],
        "scheduled_posts": []
    }
    
    try:
        result = app.invoke(initial_state)
        
        print("\n" + "=" * 60)
        print("✅ WEEKLY PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Show comprehensive results
        print(f"📊 Total Messages: {len(result['messages'])}")
        print(f"📰 Articles Processed: {len(result.get('analyzed_articles', []))}")
        print(f"🔄 Repos Processed: {len(result.get('github_data', []))}")
        print(f"🎯 Strategic Content: {len(result.get('analyzed_content', []))}")
        print(f"✍️ Posts Generated: {len(result.get('linkedin_posts', []))}")
        print(f"🔍 Posts Reviewed: {len(result.get('reviewed_posts', []))}")
        print(f"⏰ Posts Scheduled: {len(result.get('scheduled_posts', []))}")
        
        # Show scheduling summary
        scheduled_posts = result.get('scheduled_posts', [])
        if scheduled_posts:
            print(f"\n📅 SCHEDULED POSTS READY FOR DAILY PUBLISHING:")
            high_priority = [p for p in scheduled_posts if p.get('posting_priority') == 'High']
            medium_priority = [p for p in scheduled_posts if p.get('posting_priority') == 'Medium']
            
            print(f"   🔥 High Priority: {len(high_priority)} posts")
            print(f"   📈 Medium Priority: {len(medium_priority)} posts")
            print(f"   📅 Date Range: Next {len(scheduled_posts)} days")
            
            for i, post in enumerate(scheduled_posts[:3], 1):  # Show first 3
                scheduled_dt = post.get('scheduled_datetime', 'Unknown')
                try:
                    dt = datetime.fromisoformat(scheduled_dt.replace('Z', '+00:00'))
                    date_str = dt.strftime('%m/%d %H:%M')
                except:
                    date_str = 'Unknown'
                    
                print(f"   {i}. {post.get('source_title', 'Unknown')[:40]}... → {date_str}")
        
        print(f"\n🎉 Ready for daily publishing! Check your Notion database.")
        return result
        
    except Exception as e:
        print(f"\n❌ WEEKLY PIPELINE FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return None



if __name__ == "__main__":
    run_weekly_content_generation()