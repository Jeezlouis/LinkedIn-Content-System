"""
Simple Prompt Functions for LinkedIn Automation
Just copy and paste these functions into your main file
"""

def news_relevance_score(article, user_expertise, target_audience):
    """Analyzes news article relevance for LinkedIn content creation"""
    return f"""You are a professional content strategist for a software developer's LinkedIn presence. 

Analyze this news article and score its relevance for LinkedIn content creation:

Article: {article}
User Profile: {user_expertise}
User Audience: {target_audience}

Provide analysis in this format:
- Relevance Score (1-10): 
- Content Category: [AI/ML, Web Dev, Startups, DevOps, etc.]
- Technical Depth: [Beginner, Intermediate, Advanced]
- Posting Angle: [Commentary, Tutorial, News Update, Opinion]
- Key Insights: [3-5 bullet points]
- Audience Appeal: [Why this matters to the user's network]
- Trending Potential: [High/Medium/Low with reasoning]"""

def news_summarizer(article, source):
    """Extracts key information from technology news articles"""
    return f"""Extract key information from this technology news article for professional content creation:

Article: {article}
Source: {source}

Create a structured summary:
- Main Topic: [One sentence description]
- Key Points: [3-4 most important insights]
- Technical Details: [Specific technologies/methods mentioned]
- Industry Impact: [How this affects the software development industry]
- Personal Relevance Hooks: [Angles a developer could comment on]
- Related Technologies: [List relevant tech stack components]"""

def article_categorizer(title, content, source):
    """Categorizes technology content into relevant topics"""
    return f"""You are an expert at categorizing technology content for software developers.

Categorize this article into relevant topics:

Article Title: {title}
Article Content: {content}
Source: {source}

Select ALL applicable categories:
- Programming Languages (specify which: Python, JavaScript, Go, etc.)
- Frameworks & Libraries (React, Django, TensorFlow, etc.)
- Development Practices (DevOps, Testing, Architecture, etc.)
- Industry Trends (AI/ML, Blockchain, Cloud, Mobile, etc.)
- Career & Business (Startups, Leadership, Remote Work, etc.)
- Tools & Platforms (GitHub, AWS, Docker, etc.)

Also determine:
- Primary Category: [Most relevant single category]
- Content Type: [Breaking News, Analysis, Tutorial, Opinion, Research]
- Urgency Level: [Breaking/Trending/Evergreen]"""

def content_strategist(news_articles, repo_updates, past_post_metrics, days_since_last_post, recent_topics, user_skills, follower_demographics, personal_brand):
    """Makes strategic decisions about what content to create"""
    return f"""You are a LinkedIn content strategist for a software developer. Decide what content to create today.

Available Content:
- Recent News: {news_articles}
- Repository Updates: {repo_updates}
- Historical Performance: {past_post_metrics}
- Last Posted: {days_since_last_post} days ago
- Recent Post Topics: {recent_topics}

User Profile:
- Expertise: {user_skills}
- Audience: {follower_demographics}
- Brand Voice: {personal_brand}

Decision Framework:
1. Content Opportunity Score (1-10)
2. Recommended Content Type: [News Commentary, Project Showcase, Thought Leadership, Tutorial, None Today]
3. Primary Topic Focus
4. Posting Urgency: [Post Today, This Week, Save for Later]
5. Content Angle: [What unique perspective to take]
6. Reasoning: [Why this choice over alternatives]

Consider: Audience engagement patterns, topic freshness, content variety, posting frequency."""

def posting_timing_optimizer(content_type, topic, audience_segments, timezone, current_time, recent_post_times):
    """Determines optimal posting time for LinkedIn content"""
    return f"""Determine optimal posting time for this LinkedIn content:

Content Type: {content_type}
Content Topic: {topic}
Target Audience: {audience_segments}
User Timezone: {timezone}
Current Time: {current_time}
Recent Posting History: {recent_post_times}

Analysis needed:
- Optimal Posting Time: [Specific time recommendation]
- Reasoning: [Why this timing works for the audience]
- Alternative Times: [2-3 backup options]
- Posting Frequency Check: [Is this too soon after last post?]
- Weekend vs Weekday Strategy: [Day-specific considerations]

Consider: Professional audience schedules, content type engagement patterns, time zone optimization."""

def linkedin_post_writer(content_type, source_material, previous_posts, audience, content_angle, key_insights):
    """Creates LinkedIn posts optimized for professional software development audience"""
    return f"""You are a professional LinkedIn ghostwriter specializing in software development content.

Create a LinkedIn post based on:
Content Source: {content_type} - {source_material}
User Voice Sample: {previous_posts}
Target Audience: {audience}
Post Angle: {content_angle}
Key Points to Cover: {key_insights}

LinkedIn Post Requirements:
- Hook: Engaging first line that stops scrolling
- Length: 150-300 words (optimal for LinkedIn algorithm)
- Tone: Professional but conversational, matching user's voice
- Structure: Short paragraphs, easy scanning
- Call-to-Action: Encourage meaningful engagement
- Hashtags: 3-5 relevant hashtags (not excessive)
- Personal Touch: Connect to user's experience or perspective

Writing Style Guidelines:
- Start with a compelling statement or question
- Use "I" statements and personal experiences when appropriate  
- Include technical insights without being overly jargony
- End with a question or discussion prompt
- No excessive emojis or corporate speak"""

def post_variation_generator(original_concept, audience, voice_sample):
    """Creates multiple versions of LinkedIn post concept"""
    return f"""Create 3 different versions of this LinkedIn post concept:

Base Content: {original_concept}
Target Audience: {audience}
User Voice: {voice_sample}

Generate:
Version A - News Commentary: Focus on industry analysis and implications
Version B - Personal Experience: Connect to user's projects and learnings  
Version C - Community Discussion: Emphasize questions and engagement

Each version should:
- Maintain the core message
- Use different hooks and structures
- Appeal to different engagement styles
- Stay true to the user's authentic voice
- Be 150-300 words"""

def content_reviewer(post_draft, brand_voice, audience, topic):
    """Reviews LinkedIn content for quality and professional standards"""
    return f"""You are a LinkedIn content quality auditor. Review this post for professional standards:

Post Content: {post_draft}
User Brand Guidelines: {brand_voice}
Target Audience: {audience}
Content Topic: {topic}

Quality Assessment:
- Brand Alignment (1-10): [Does this match the user's professional image?]
- Engagement Potential (1-10): [Likely to generate meaningful comments/shares?]
- Technical Accuracy: [Any factual errors or questionable claims?]
- Tone Appropriateness: [Professional yet authentic?]
- LinkedIn Optimization: [Hook strength, length, readability?]
- Improvement Suggestions: [Specific recommendations]
- Approval Status: [Approve/Revise/Reject with reasoning]

Red Flags to Check:
- Overly promotional or salesy language
- Controversial statements without context
- Technical inaccuracies
- Off-brand voice or tone
- Poor grammar or formatting"""

def engagement_predictor(post_text, past_engagement_data, content_category, scheduled_time, trending_topics):
    """Predicts engagement potential for LinkedIn posts"""
    return f"""Predict engagement potential for this LinkedIn post:

Post Content: {post_text}
User's Historical Performance: {past_engagement_data}
Topic Category: {content_category}
Posting Time: {scheduled_time}
Current Industry Trends: {trending_topics}

Engagement Prediction:
- Expected Likes: [Range estimate]
- Expected Comments: [Range estimate]
- Expected Shares: [Range estimate]
- Viral Potential: [Low/Medium/High]
- Best Performing Elements: [What makes this engaging?]
- Risk Factors: [What might limit engagement?]
- Optimization Suggestions: [How to improve performance]

Base prediction on: Content quality, topic relevance, timing, historical patterns."""

def repo_significance_analyzer(repo_name, commit_messages, diff_summary, repo_description, tech_stack):
    """Analyzes repository changes for LinkedIn content potential"""
    return f"""Analyze these repository changes for LinkedIn content potential:

Repository: {repo_name}
Recent Commits: {commit_messages}
Code Changes: {diff_summary}
Project Description: {repo_description}
Technologies Used: {tech_stack}

Analysis:
- Significance Level (1-10): [How noteworthy are these changes?]
- Content Angle: [Project Showcase, Technical Tutorial, Learning Journey, etc.]
- Story Potential: [What narrative could this tell?]
- Technical Insights: [What could others learn from this?]
- Professional Value: [How does this demonstrate skills/growth?]
- Audience Interest: [Who would find this relevant?]
- Content Recommendation: [Should this become a LinkedIn post?]

Focus on: New features, architectural decisions, problem-solving approaches, learning experiences."""

def performance_analyzer(post_text, engagement_metrics, comment_themes, posting_details, comparison_data):
    """Analyzes LinkedIn post performance for future optimization"""
    return f"""Analyze LinkedIn post performance to improve future content strategy:

Post Content: {post_text}
Engagement Metrics: {engagement_metrics}
Audience Response: {comment_themes}
Posting Details: {posting_details}
Historical Context: {comparison_data}

Performance Analysis:
- Success Factors: [What worked well?]
- Underperforming Elements: [What could be improved?]
- Audience Insights: [What does engagement reveal about followers?]
- Content Lessons: [Key takeaways for future posts]
- Strategy Adjustments: [How should future content change?]
- Topic Performance: [How did this topic perform vs others?]
- Timing Insights: [Was posting time optimal?]

Generate actionable insights for improving content strategy and audience engagement."""

def news_extractor(scraped_content, source_url, timestamp):
    """Extracts structured information from web-scraped technology news"""
    return f"""You are an expert at extracting structured information from web-scraped technology news content.

Raw scraped content: {scraped_content}
Source website: {source_url}
Scrape timestamp: {timestamp}

Extract the following information:

ARTICLES:
For each distinct news article found, provide:
- Title: [Clean article headline]
- Summary: [2-3 sentence summary of the main points]  
- Publication Date: [When was this published - extract from content]
- Author: [Article author if available]
- Article URL: [Direct link to full article if found]
- Main Topic: [Primary subject matter]
- Key Technologies Mentioned: [Specific tools, languages, frameworks, companies]

CONTENT CLEANING:
- Remove navigation menus, ads, cookie notices, footer content
- Ignore comments sections and related article suggestions
- Focus only on actual article content
- Combine article fragments if they appear to be from the same story

OUTPUT FORMAT:
Return a JSON-like structure for each article:
```json
{{
  "title": "...",
  "summary": "...", 
  "publication_date": "...",
  "author": "...",
  "article_url": "...",
  "main_topic": "...",
  "technologies": ["...", "..."],
  "content": "First paragraph or two..."
}}
```

If content is unclear or fragmented, mark with "NEEDS_REVIEW" flag."""