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

Return ONLY valid JSON in this exact format:
{{
    "relevance_score": score_1_to_10,
    "content_category": "AI/ML|Web Dev|Startups|DevOps|etc",
    "technical_depth": "Beginner|Intermediate|Advanced",
    "posting_angle": "Commentary|Tutorial|News Update|Opinion",
    "key_insights": ["insight 1", "insight 2", "insight 3"],
    "audience_appeal": "why this matters to the user's network",
    "trending_potential": {{
        "level": "High|Medium|Low",
        "reasoning": "explanation for trending potential"
    }}
}}
"""

def news_summarizer(article, source):
    """Extracts key information from technology news articles"""
    return f"""Extract key information from this technology news article for professional content creation:

Article: {article}
Source: {source}

Return ONLY valid JSON in this exact format:
{{
    "main_topic": "One sentence description",
    "key_points": ["point 1", "point 2", "point 3", "point 4"],
    "technical_details": ["specific technologies/methods mentioned"],
    "industry_impact": "How this affects the software development industry",
    "personal_relevance_hooks": ["angle 1", "angle 2", "angle 3"],
    "related_technologies": ["tech 1", "tech 2", "tech 3"]
}}
"""

def article_categorizer(title, content, source):
    """Categorizes technology content into relevant topics"""
    return f"""You are an expert at categorizing technology content for software developers.

Categorize this article into relevant topics:

Article Title: {title}
Article Content: {content}
Source: {source}

Return ONLY valid JSON in this exact format:
{{
    "applicable_categories": {{
        "programming_languages": ["Python", "JavaScript", "Go", "etc"],
        "frameworks_libraries": ["React", "Django", "TensorFlow", "etc"],
        "development_practices": ["DevOps", "Testing", "Architecture", "etc"],
        "industry_trends": ["AI/ML", "Blockchain", "Cloud", "Mobile", "etc"],
        "career_business": ["Startups", "Leadership", "Remote Work", "etc"],
        "tools_platforms": ["GitHub", "AWS", "Docker", "etc"]
    }},
    "primary_category": "Most relevant single category",
    "content_type": "Breaking News|Analysis|Tutorial|Opinion|Research",
    "urgency_level": "Breaking|Trending|Evergreen"
}}
"""

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

Return ONLY valid JSON in this exact format:
{{
    "content_opportunity_score": score_1_to_10,
    "recommended_content_type": "News Commentary|Project Showcase|Thought Leadership|Tutorial|None Today",
    "primary_topic_focus": "specific topic to focus on",
    "posting_urgency": "Post Today|This Week|Save for Later",
    "content_angle": "What unique perspective to take",
    "reasoning": "Why this choice over alternatives",
    "audience_engagement_considerations": "engagement patterns analysis",
    "topic_freshness_score": score_1_to_10,
    "content_variety_balance": "how this fits with recent posts"
}}

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

Return ONLY valid JSON in this exact format:
{{
    "optimal_posting_time": {{
        "recommended_datetime": "specific time recommendation",
        "reasoning": "Why this timing works for the audience"
    }},
    "alternative_times": [
        {{
            "datetime": "alternative time 1",
            "effectiveness_score": score_1_to_10
        }},
        {{
            "datetime": "alternative time 2", 
            "effectiveness_score": score_1_to_10
        }}
    ],
    "posting_frequency_check": {{
        "too_soon": true_or_false,
        "optimal_gap": "recommended time since last post",
        "frequency_analysis": "current posting pattern analysis"
    }},
    "day_strategy": {{
        "weekend_vs_weekday": "day-specific considerations",
        "audience_schedule_alignment": "professional audience timing factors"
    }}
}}

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

Return ONLY valid JSON in this exact format:
{{
    "post_content": "complete LinkedIn post text (150-300 words)",
    "post_structure": {{
        "hook": "engaging first line that stops scrolling",
        "body": "main content with short paragraphs",
        "call_to_action": "question or discussion prompt"
    }},
    "tone_analysis": "professional but conversational voice",
    "hashtags": ["hashtag1", "hashtag2", "hashtag3"],
    "personal_touch": "how this connects to user's experience",
    "engagement_optimization": {{
        "question_included": true_or_false,
        "personal_experience_shared": true_or_false,
        "technical_insights_provided": true_or_false
    }},
    "media_suggestions": {{
        "image_recommended": true_or_false,
        "image_description": "description of what image would enhance this post",
        "visual_elements": ["screenshot", "diagram", "code snippet", "etc"]
    }}
}}

Requirements: Hook that stops scrolling, 150-300 words, professional but conversational tone, short paragraphs, personal connection, 3-5 hashtags, meaningful engagement prompt."""

def post_variation_generator(original_concept, audience, voice_sample):
    """Creates multiple versions of LinkedIn post concept"""
    return f"""Create 3 different versions of this LinkedIn post concept:

Base Content: {original_concept}
Target Audience: {audience}
User Voice: {voice_sample}

Return ONLY valid JSON in this exact format:
{{
    "variations": {{
        "version_a_news_commentary": {{
            "focus": "industry analysis and implications",
            "content": "complete post focusing on news commentary (150-300 words)",
            "engagement_style": "analytical discussion"
        }},
        "version_b_personal_experience": {{
            "focus": "connect to user's projects and learnings",
            "content": "complete post with personal connection (150-300 words)",
            "engagement_style": "experience sharing"
        }},
        "version_c_community_discussion": {{
            "focus": "emphasize questions and engagement",
            "content": "complete post focused on community discussion (150-300 words)",
            "engagement_style": "question-driven"
        }}
    }},
    "common_elements": {{
        "core_message": "shared message across all versions",
        "user_voice_consistency": "how all maintain authentic voice",
        "target_length": "150-300 words for each"
    }}
}}

Each version should maintain the core message, use different hooks and structures, appeal to different engagement styles, and stay true to the user's authentic voice."""


def content_reviewer(post_draft, brand_voice, audience, topic):
    """Reviews LinkedIn content for quality and professional standards"""
    return f"""You are a LinkedIn content quality auditor. Review this post for professional standards:

Post Content: {post_draft}
User Brand Guidelines: {brand_voice}
Target Audience: {audience}
Content Topic: {topic}

Return ONLY valid JSON in this exact format:
{{
    "quality_assessment": {{
        "brand_alignment": {{
            "score": score_1_to_10,
            "analysis": "Does this match the user's professional image?"
        }},
        "engagement_potential": {{
            "score": score_1_to_10,
            "analysis": "Likely to generate meaningful comments/shares?"
        }},
        "technical_accuracy": {{
            "status": "accurate|needs_verification|contains_errors",
            "notes": "Any factual errors or questionable claims?"
        }},
        "tone_appropriateness": {{
            "status": "appropriate|needs_adjustment",
            "analysis": "Professional yet authentic?"
        }},
        "linkedin_optimization": {{
            "hook_strength": score_1_to_10,
            "length_optimal": true_or_false,
            "readability": score_1_to_10
        }}
    }},
    "improvement_suggestions": ["specific recommendation 1", "specific recommendation 2"],
    "approval_status": "Approve|Revise|Reject",
    "reasoning": "detailed explanation for approval status",
    "red_flags": {{
        "promotional_language": true_or_false,
        "controversial_statements": true_or_false,
        "technical_inaccuracies": true_or_false,
        "off_brand_voice": true_or_false,
        "grammar_formatting_issues": true_or_false
    }}
}}

Check for: Overly promotional language, controversial statements without context, technical inaccuracies, off-brand voice, poor grammar or formatting."""

def engagement_predictor(post_text, past_engagement_data, content_category, scheduled_time, trending_topics):
    """Predicts engagement potential for LinkedIn posts"""
    return f"""Predict engagement potential for this LinkedIn post:

Post Content: {post_text}
User's Historical Performance: {past_engagement_data}
Topic Category: {content_category}
Posting Time: {scheduled_time}
Current Industry Trends: {trending_topics}

Return ONLY valid JSON in this exact format:
{{
    "engagement_predictions": {{
        "expected_likes": {{
            "range_min": number,
            "range_max": number,
            "confidence": "high|medium|low"
        }},
        "expected_comments": {{
            "range_min": number,
            "range_max": number,
            "confidence": "high|medium|low"
        }},
        "expected_shares": {{
            "range_min": number,
            "range_max": number,
            "confidence": "high|medium|low"
        }},
        "viral_potential": "Low|Medium|High"
    }},
    "performance_factors": {{
        "best_performing_elements": ["element 1", "element 2"],
        "risk_factors": ["risk 1", "risk 2"],
        "optimization_suggestions": ["suggestion 1", "suggestion 2"]
    }},
    "historical_comparison": {{
        "vs_user_average": "above|average|below",
        "similar_content_performance": "better|similar|worse"
    }}
}}

Base prediction on: Content quality, topic relevance, timing, historical patterns."""


def repo_significance_analyzer(repo_name, commit_messages, diff_summary, repo_description, repo_content_summary, tech_stack, images=None):
    """Analyzes repository changes for LinkedIn content potential"""
    
    image_context = ""
    if images and images.get('readme_images'):
        image_context = f"\nAvailable Images: {images['readme_images'][:3]}"  # Show first 3 images
    elif images and images.get('primary_image'):
        image_context = f"\nPrimary Image: {images['primary_image']}"
    
    return f"""Analyze these repository changes for LinkedIn content potential:

Repository: {repo_name}
Recent Commits: {commit_messages}
Code Changes: {diff_summary}
Project Description: {repo_description}
Project Content Summary: {repo_content_summary}
Technologies Used: {tech_stack}{image_context}

OUTPUT FORMAT:
Return ONLY valid JSON (no explanations, no markdown). The JSON should look like this:

{{
  "significance_level": between 1 - 10,
  "content_angles": ["Project Showcase", "Technical Tutorial", "Learning Journey"],
  "story_potential": {{
    "problem_solution": "string",
    "tech_highlight": "string"
  }},
  "technical_insights": ["string", "string"],
  "professional_value": ["string", "string"],
  "technologies": ["string", "string"],
  "audience_interest": {{
    "developers": "string",
    "job_seekers": "string",
    "product_managers": "string",
    "tech_leads": "string"
  }},
  "content_recommendations": {{
    "verdict": "string",
    "top_formats": ["string", "string"]
  }},
  "content_summary": "The summary of the entire project in 2 - 3 sentences",
  "diff_summary": "string",
  "key_hook": "string",
  "hashtags": ["string", "string"],
  "visual_content": {{
    "has_images": true_or_false,
    "image_urls": ["url1", "url2"],
    "visual_appeal": "description of visual elements that would enhance LinkedIn post"
  }}
}}

Focus on: New features, architectural decisions, problem-solving approaches, learning experiences."""


def performance_analyzer(post_text, engagement_metrics, comment_themes, posting_details, comparison_data):
    """Analyzes LinkedIn post performance for future optimization"""
    return f"""Analyze LinkedIn post performance to improve future content strategy:

Post Content: {post_text}
Engagement Metrics: {engagement_metrics}
Audience Response: {comment_themes}
Posting Details: {posting_details}
Historical Context: {comparison_data}

Return ONLY valid JSON in this exact format:
{{
    "performance_summary": {{
        "overall_success_level": "excellent|good|average|poor",
        "exceeded_expectations": true_or_false
    }},
    "success_factors": ["what worked well 1", "what worked well 2"],
    "underperforming_elements": ["what could be improved 1", "what could be improved 2"],
    "audience_insights": {{
        "engagement_quality": "meaningful professional discussions vs surface engagement",
        "follower_behavior_patterns": "what the engagement reveals about followers"
    }},
    "content_lessons": ["key takeaway 1", "key takeaway 2"],
    "strategy_adjustments": ["how future content should change 1", "how future content should change 2"],
    "topic_performance": {{
        "vs_other_topics": "better|similar|worse",
        "topic_specific_insights": "how this topic performed"
    }},
    "timing_insights": {{
        "posting_time_effectiveness": "optimal|suboptimal",
        "timing_recommendations": "future timing suggestions"
    }}
}}

Generate actionable insights for improving content strategy and audience engagement."""

def news_extractor(scraped_content, source_url, timestamp):
    """Extracts structured information from web-scraped technology news - SIMPLIFIED"""
    return f"""You are an expert at extracting structured information from web-scraped technology news content.

Raw scraped content: {scraped_content}
Source website: {source_url}
Scrape timestamp: {timestamp}

INSTRUCTIONS:
1. Extract all distinct news articles from the content
2. Look for image URLs in the scraped content (img src, featured images, article thumbnails)
3. Return ONLY valid JSON - no explanations, no markdown blocks
4. If information is missing, use these defaults:
   - Author: "TLDR"
   - Article URL: "{source_url}"
   - Image URL: "" (but try to find actual image URLs from the content)
   - Source: "TLDR"

REQUIRED OUTPUT FORMAT (return exactly this structure):

[
  {{
    "title": "Article headline here",
    "summary": "Brief 2-3 sentence summary",
    "publication_date": "2024-01-01",
    "author": "string",
    "article_url": "{source_url}",
    "image_url": "https://example.com/article-image.jpg (extract from img tags, featured images, or thumbnails in the content)",
    "source": "string",
    "main_topic": "Technology",
    "technologies": ["tech1", "tech2"],
    "content": "First paragraph of article content"
  }}
]

Return only the JSON array. No other text."""