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
from prompt import news_extractor, news_summarizer, news_relevance_score, article_categorizer, repo_significance_analyzer
import os
import json, re
import time
import traceback

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
    github_data: List[dict] 



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
            print("   Raw LLM response (first 300 chars):", response.content[:300])

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

def save_repo_to_notion(state: AgentState) -> AgentState:
    """Node 6: Save analyzed repo activites to Notion database"""
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

# Define the flow - FIXED VERSION
# graph.add_edge(START, "scraper")
# graph.add_edge("scraper", "extractor") 
# graph.add_edge("extractor", "relevance_analyst")
# graph.add_edge("relevance_analyst", "categorizer")
# graph.add_edge("categorizer", "summarizer")  # ← THIS WAS MISSING!
# graph.add_edge("summarizer", "notion_news_saver")
# graph.add_edge("notion_news_saver", END)

# to test the github analyzer alone
graph.add_edge(START, "github_analyzer")
graph.add_edge("github_analyzer", "notion_github_saver")
graph.add_edge("notion_github_saver", END)

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
        "github_data": [],
    }
    
    try:
        result = app.invoke(initial_state)
        
        print("\n" + "=" * 50)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"📊 Total Messages: {len(result['messages'])}")
        print(f"📰 Repos Processed: {len(result.get('github_data', []))}")
        
        # Show complete JSON content for debugging
        # for i, repo in enumerate(result.get('github_data', []), 1):
        #     print(f"\n" + "="*60)
        #     print(f"📄 REPO {i} - COMPLETE JSON:")
        #     print("="*60)
            
            # try:
            #     formatted_json = json.dumps(repo, indent=2, ensure_ascii=False)
            #     print(formatted_json)
            # except Exception as e:
            #     print(f"JSON formatting error: {e}")
            #     print("Raw repo data:", repo)
            
            # print("="*60)
        
        return result
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_news_analysis()