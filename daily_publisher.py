from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START, END
from typing import List, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from notion_client import Client
from datetime import datetime, timezone, timedelta
import os
import json
import time
import schedule
from prompt import performance_analyzer

load_dotenv()

class DailyAgentState(TypedDict):
    messages: List[BaseMessage]
    scheduled_posts: List[dict]
    todays_post: dict
    published_post: dict
    performance_data: dict

# Initialize LLM
llm = ChatDeepSeek(model="deepseek-chat")

# Improved query logic for daily publisher
def fetch_scheduled_posts_for_today(state: DailyAgentState) -> DailyAgentState:
    """Node 1: Fetch posts scheduled for today from Notion - IMPROVED VERSION"""
    print("📅 STEP 1: Fetching posts scheduled for today...")
    
    try:
        notion = Client(auth=os.getenv("NOTION_TOKEN"))
        database_id = os.getenv("LINKEDIN_POSTS_DATABASE_ID")
        
        today = datetime.now().date().isoformat()
        current_hour = datetime.now().hour
        
        # More flexible query - get all posts for today that haven't been published
        response = notion.databases.query(
            database_id=database_id,
            filter={
                "and": [
                    {
                        "property": "Post Status",
                        "status": {"equals": "Scheduled"}
                    },
                    {
                        "property": "Scheduled Date",
                        "date": {"equals": today}
                    },
                    {
                        "property": "LinkedIn Post Created", 
                        "checkbox": {"equals": False}
                    },
                    {
                        "property": "Ready for Publishing",
                        "checkbox": {"equals": True}
                    }
                ]
            },
            sorts=[
                {
                    "property": "Posting Priority",
                    "direction": "descending"
                },
                {
                    "property": "Content Quality Score",
                    "direction": "descending"
                },
                {
                    "property": "Scheduled Time",  # Add time-based sorting
                    "direction": "ascending"
                }
            ]
        )
        
        scheduled_posts = []
        for page in response["results"]:
            properties = page["properties"]
            
            # Get scheduled time
            scheduled_time_text = properties.get("Scheduled Time", {}).get("rich_text", [{}])[0].get("text", {}).get("content", "14:00")
            
            # Parse scheduled time
            try:
                scheduled_hour = int(scheduled_time_text.split(":")[0])
            except:
                scheduled_hour = 14  # Default to 2 PM
            
            # IMPROVED LOGIC: More flexible time window
            time_diff = abs(current_hour - scheduled_hour)
            
            # Include posts if:
            # 1. Exact hour match, OR
            # 2. Within 2 hours and high priority, OR  
            # 3. Past due (missed posts), OR
            # 4. No other posts found and it's after 9 AM
            priority = properties.get("Posting Priority", {}).get("select", {}).get("name", "Medium")
            
            should_include = (
                time_diff == 0 or  # Exact time
                (time_diff <= 2 and priority == "High") or  # High priority with wider window
                (current_hour > scheduled_hour and current_hour - scheduled_hour <= 6) or  # Past due but not too old
                (len(scheduled_posts) == 0 and current_hour >= 9)  # Fallback for morning posts
            )
            
            if should_include:
                post_data = {
                    "notion_page_id": page["id"],
                    "title": properties.get("Post Title", {}).get("title", [{}])[0].get("text", {}).get("content", ""),
                    "content": properties.get("Post Content", {}).get("rich_text", [{}])[0].get("text", {}).get("content", ""),
                    "content_type": properties.get("Content Type", {}).get("select", {}).get("name", ""),
                    "posting_priority": priority,
                    "quality_score": properties.get("Content Quality Score", {}).get("number", 5),
                    "scheduled_time": scheduled_time_text,
                    "scheduled_date": today,
                    "approval_status": properties.get("Approval Status", {}).get("select", {}).get("name", "Approve"),
                    "time_difference": time_diff,  # Track how far off we are
                    "variations": {
                        "a": properties.get("Variation A - News Commentary", {}).get("rich_text", [{}])[0].get("text", {}).get("content", ""),
                        "b": properties.get("Variation B - Personal Experience", {}).get("rich_text", [{}])[0].get("text", {}).get("content", ""),
                        "c": properties.get("Variation C - Community Discussion", {}).get("rich_text", [{}])[0].get("text", {}).get("content", "")
                    }
                }
                scheduled_posts.append(post_data)
        
        print(f"📊 Found {len(scheduled_posts)} posts ready for publishing")
        
        return {
            "scheduled_posts": scheduled_posts,
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Found {len(scheduled_posts)} posts ready for publishing today")
            ]
        }
        
    except Exception as e:
        print(f"❌ Failed to fetch scheduled posts: {e}")
        return {
            "scheduled_posts": [],
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Error fetching scheduled posts: {str(e)}")
            ]
        }

        
def select_todays_post(state: DailyAgentState) -> DailyAgentState:
    """Node 2: Select the best post to publish right now"""
    print("🎯 STEP 2: Selecting best post for immediate publishing...")
    
    scheduled_posts = state.get("scheduled_posts", [])
    
    if not scheduled_posts:
        print("ℹ️ No posts scheduled for this time slot")
        return {
            "todays_post": {},
            "messages": state.get("messages", []) + [
                SystemMessage(content="No posts scheduled for publishing at this time")
            ]
        }
    
    # Simple selection logic: Priority > Quality Score > Approval Status
    best_post = None
    best_score = -1
    
    for post in scheduled_posts:
        score = 0
        
        # Priority scoring
        priority = post.get("posting_priority", "Medium")
        if priority == "High":
            score += 10
        elif priority == "Medium":
            score += 5
        
        # Quality score
        score += post.get("quality_score", 5)
        
        # Approval bonus
        if post.get("approval_status") == "Approve":
            score += 2
        
        if score > best_score:
            best_score = score
            best_post = post
    
    if best_post:
        print(f"✅ Selected: {best_post.get('title', 'Unknown')[:50]}...")
        print(f"📊 Priority: {best_post.get('posting_priority')}, Quality: {best_post.get('quality_score')}/10")
    else:
        print("⚠️ No suitable post found for publishing")
    
    return {
        "todays_post": best_post or {},
        "messages": state.get("messages", []) + [
            SystemMessage(content=f"Selected post for publishing: {best_post.get('title', 'None') if best_post else 'None'}")
        ]
    }

def publish_to_linkedin(state: DailyAgentState) -> DailyAgentState:
    """Node 3: Publish the selected post to LinkedIn"""
    print("🚀 STEP 3: Publishing to LinkedIn...")
    
    todays_post = state.get("todays_post", {})
    
    if not todays_post:
        print("ℹ️ No post selected for publishing")
        return {
            "published_post": {},
            "messages": state.get("messages", []) + [
                SystemMessage(content="No post to publish today")
            ]
        }
    
    # TODO: Replace with actual LinkedIn API integration
    print("📝 SIMULATING LinkedIn publishing...")
    print(f"🎯 Title: {todays_post.get('title', 'Unknown')}")
    print(f"📊 Content Preview: {todays_post.get('content', '')[:150]}...")
    print(f"🏷️ Priority: {todays_post.get('posting_priority')}")
    
    # Simulate publishing delay
    time.sleep(2)
    
    # Update Notion to mark as published
    try:
        notion = Client(auth=os.getenv("NOTION_TOKEN"))
        notion_page_id = todays_post.get("notion_page_id")
        
        if notion_page_id:
            notion.pages.update(
                page_id=notion_page_id,
                properties={
                    "LinkedIn Post Created": {"checkbox": True},
                    "Post Status": {"status": {"name": "Published"}},
                    "Published Date": {"date": {"start": datetime.now().date().isoformat()}},
                    "Published Time": {"rich_text": [{"text": {"content": datetime.now().strftime("%H:%M")}}]},
                    "Publication Notes": {"rich_text": [{"text": {"content": f"Published successfully at {datetime.now().isoformat()}"}}]}
                }
            )
            print("✅ Updated Notion: Post marked as published")
        
    except Exception as e:
        print(f"⚠️ Failed to update Notion: {e}")
    
    # Create published post record
    published_post = {
        "linkedin_post_id": f"linkedin_post_{int(time.time())}",  # Would be real LinkedIn ID
        "published_at": datetime.now().isoformat(),
        "post_content": todays_post.get("content", ""),
        "post_title": todays_post.get("title", ""),
        "content_type": todays_post.get("content_type", ""),
        "posting_priority": todays_post.get("posting_priority", ""),
        "quality_score": todays_post.get("quality_score", 5),
        "notion_page_id": todays_post.get("notion_page_id"),
        "scheduled_time": todays_post.get("scheduled_time", ""),
        "actual_publish_time": datetime.now().strftime("%H:%M")
    }
    
    print("🎉 Post successfully published to LinkedIn!")
    
    return {
        "published_post": published_post,
        "messages": state.get("messages", []) + [
            SystemMessage(content=f"Successfully published: {todays_post.get('title', 'Unknown post')}")
        ]
    }

def track_performance(state: DailyAgentState) -> DailyAgentState:
    """Node 4: Track post performance (initial metrics + setup monitoring)"""
    print("📊 STEP 4: Setting up performance tracking...")
    
    published_post = state.get("published_post", {})
    
    if not published_post:
        print("ℹ️ No published post to track")
        return {
            "performance_data": {},
            "messages": state.get("messages", []) + [
                SystemMessage(content="No post performance to track")
            ]
        }
    
    # TODO: Replace with actual LinkedIn API metrics
    print("📈 SIMULATING performance tracking setup...")
    print(f"🎯 Tracking post: {published_post.get('post_title', 'Unknown')}")
    
    # Simulate initial performance data
    performance_data = {
        "linkedin_post_id": published_post.get("linkedin_post_id"),
        "initial_metrics": {
            "likes": 0,
            "comments": 0,
            "shares": 0,
            "views": 0,
            "clicks": 0
        },
        "tracking_started_at": datetime.now().isoformat(),
        "next_check_at": (datetime.now() + timedelta(hours=1)).isoformat(),
        "post_details": {
            "title": published_post.get("post_title", ""),
            "content_type": published_post.get("content_type", ""),
            "quality_score": published_post.get("quality_score", 5),
            "posting_priority": published_post.get("posting_priority", "")
        },
        "tracking_schedule": [
            "1 hour", "6 hours", "24 hours", "3 days", "1 week"
        ]
    }
    
    # Update Notion with performance tracking info
    try:
        notion = Client(auth=os.getenv("NOTION_TOKEN"))
        notion_page_id = published_post.get("notion_page_id")
        
        if notion_page_id:
            notion.pages.update(
                page_id=notion_page_id,
                properties={
                    "Performance Tracking": {"checkbox": True},
                    "Tracking Started": {"date": {"start": datetime.now().date().isoformat()}},
                    "Initial Likes": {"number": 0},
                    "Initial Comments": {"number": 0},
                    "Initial Shares": {"number": 0}
                }
            )
            print("✅ Updated Notion: Performance tracking initialized")
        
    except Exception as e:
        print(f"⚠️ Failed to initialize performance tracking in Notion: {e}")
    
    print("📊 Performance tracking setup completed")
    
    return {
        "performance_data": performance_data,
        "messages": state.get("messages", []) + [
            SystemMessage(content="Performance tracking initialized for published post")
        ]
    }

def analyze_performance_and_learn(state: DailyAgentState) -> DailyAgentState:
    """Node 5: Analyze performance data and extract learning insights"""
    print("🧠 STEP 5: Performance analysis and learning...")
    
    published_post = state.get("published_post", {})
    performance_data = state.get("performance_data", {})
    
    if not published_post or not performance_data:
        print("ℹ️ No performance data to analyze yet")
        return {
            "messages": state.get("messages", []) + [
                SystemMessage(content="No performance data available for analysis")
            ]
        }
    
    try:
        # For now, simulate performance analysis
        # TODO: Replace with actual LinkedIn API metrics after some time has passed
        
        # Simulate some basic metrics for demonstration
        simulated_metrics = {
            "likes": 5,  # Would be real data from LinkedIn API
            "comments": 2,
            "shares": 1,
            "views": 100,
            "clicks": 10,
            "engagement_rate": 0.18  # (likes + comments + shares) / views
        }
        
        post_content = published_post.get("post_content", "")
        content_type = published_post.get("content_type", "")
        posting_priority = published_post.get("posting_priority", "")
        
        # Use your existing performance analyzer prompt
        system_prompt = performance_analyzer(
            post_text=post_content,
            engagement_metrics=simulated_metrics,
            comment_themes=["positive feedback", "questions about implementation"],
            posting_details={
                "content_type": content_type,
                "priority": posting_priority,
                "scheduled_time": published_post.get("scheduled_time", ""),
                "actual_time": published_post.get("actual_publish_time", "")
            },
            comparison_data={"average_likes": 8, "average_comments": 3, "average_shares": 1}
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Analyze this post's performance and provide learning insights.")
        ]
        
        response = llm.invoke(messages)
        
        # Parse the analysis
        import re
        clean_content = re.sub(r"```(?:json)?|```", "", response.content).strip()
        analysis_result = json.loads(clean_content)
        
        # Extract key insights
        success_level = analysis_result.get("performance_summary", {}).get("overall_success_level", "average")
        success_factors = analysis_result.get("success_factors", [])
        content_lessons = analysis_result.get("content_lessons", [])
        
        print(f"📊 Performance Analysis: {success_level.upper()}")
        print(f"✅ Success Factors: {', '.join(success_factors[:2])}")
        print(f"📚 Key Lessons: {', '.join(content_lessons[:2])}")
        
        # Update Notion with analysis
        try:
            notion = Client(auth=os.getenv("NOTION_TOKEN"))
            notion_page_id = published_post.get("notion_page_id")
            
            if notion_page_id:
                notion.pages.update(
                    page_id=notion_page_id,
                    properties={
                        "Performance Analysis": {"rich_text": [{"text": {"content": f"Success Level: {success_level}"}}]},
                        "Success Factors": {"rich_text": [{"text": {"content": "; ".join(success_factors)[:1900]}}]},
                        "Learning Insights": {"rich_text": [{"text": {"content": "; ".join(content_lessons)[:1900]}}]},
                        "Engagement Rate": {"number": simulated_metrics.get("engagement_rate", 0) * 100},
                        "Final Likes": {"number": simulated_metrics.get("likes", 0)},
                        "Final Comments": {"number": simulated_metrics.get("comments", 0)},
                        "Final Shares": {"number": simulated_metrics.get("shares", 0)}
                    }
                )
                print("✅ Updated Notion: Performance analysis saved")
            
        except Exception as e:
            print(f"⚠️ Failed to save analysis to Notion: {e}")
        
        return {
            "performance_data": {
                **performance_data,
                "analysis": analysis_result,
                "metrics": simulated_metrics,
                "analyzed_at": datetime.now().isoformat()
            },
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Performance analysis completed: {success_level} performance with key insights extracted")
            ]
        }
        
    except Exception as e:
        print(f"❌ Performance analysis failed: {e}")
        return {
            "performance_data": performance_data,
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Performance analysis failed: {str(e)}")
            ]
        }

# Build the daily publishing graph
daily_graph = StateGraph(DailyAgentState)

# Add nodes
daily_graph.add_node("fetch_scheduled", fetch_scheduled_posts_for_today)
daily_graph.add_node("select_post", select_todays_post)
daily_graph.add_node("publish_post", publish_to_linkedin)
daily_graph.add_node("track_performance", track_performance)
daily_graph.add_node("analyze_performance", analyze_performance_and_learn)

# Define the flow
daily_graph.add_edge(START, "fetch_scheduled")
daily_graph.add_edge("fetch_scheduled", "select_post")
daily_graph.add_edge("select_post", "publish_post")
daily_graph.add_edge("publish_post", "track_performance")
daily_graph.add_edge("track_performance", "analyze_performance")
daily_graph.add_edge("analyze_performance", END)

# Compile the daily graph
daily_app = daily_graph.compile()

def run_daily_publishing():
    """Run the daily publishing and performance tracking pipeline"""
    print("📅 DAILY PUBLISHING & PERFORMANCE PIPELINE")
    print("⚡ Publishing → Tracking → Learning")
    print("=" * 60)
    
    initial_state = {
        "messages": [],
        "scheduled_posts": [],
        "todays_post": {},
        "published_post": {},
        "performance_data": {}
    }
    
    try:
        result = daily_app.invoke(initial_state)
        
        print("\n" + "=" * 60)
        print("✅ DAILY PUBLISHING COMPLETED!")
        print("=" * 60)
        
        published_post = result.get("published_post", {})
        performance_data = result.get("performance_data", {})
        
        if published_post:
            print(f"🚀 Published: {published_post.get('post_title', 'Unknown')}")
            print(f"⏰ Time: {published_post.get('actual_publish_time', 'Unknown')} (Scheduled: {published_post.get('scheduled_time', 'Unknown')})")
            print(f"📊 Priority: {published_post.get('posting_priority', 'Unknown')}")
            print(f"🎯 Quality Score: {published_post.get('quality_score', 'Unknown')}/10")
            
            if performance_data.get("analysis"):
                analysis = performance_data["analysis"]
                success_level = analysis.get("performance_summary", {}).get("overall_success_level", "Unknown")
                print(f"📈 Performance: {success_level.upper()}")
        else:
            print("ℹ️ No post was published today (none scheduled for this time)")
        
        return result
        
    except Exception as e:
        print(f"❌ DAILY PUBLISHING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

# Scheduling functions
def schedule_daily_publishing():
    """Schedule daily publishing at multiple optimal times"""
    
    # Schedule publishing at professional peak times (UTC)
    schedule.every().day.at("09:00").do(run_daily_publishing)  # 9 AM - Morning professionals
    schedule.every().day.at("13:00").do(run_daily_publishing)  # 1 PM - Lunch break engagement
    schedule.every().day.at("17:00").do(run_daily_publishing)  # 5 PM - End of workday
    
    print("📅 Daily publishing scheduled for 9 AM, 1 PM, and 5 PM UTC")
    print("⏰ Each run checks for posts scheduled at that time")
    
    while True:
        schedule.run_pending()
        time.sleep(300)  # Check every 5 minutes

if __name__ == "__main__":
    # For testing - run immediately
    print("🧪 RUNNING DAILY PIPELINE TEST")
    run_daily_publishing()
    
    # For production - run scheduler
    # schedule_daily_publishing()