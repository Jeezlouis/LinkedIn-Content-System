# 🚀 LinkedIn Content Management System

An intelligent, automated LinkedIn content creation and publishing system that transforms your GitHub activity and tech news into engaging professional posts.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Integrations](#api-integrations)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

This system automates your LinkedIn content strategy by:

1. **Monitoring** your GitHub repositories for recent activity
2. **Scraping** technology news from multiple sources
3. **Analyzing** content relevance using AI
4. **Generating** professional LinkedIn posts with multiple variations
5. **Scheduling** and **publishing** content automatically
6. **Tracking** performance and learning from engagement

## ✨ Features

### 🤖 Intelligent Content Generation
- **Multi-source news aggregation** from TLDR Tech, TechCrunch, Hacker News
- **GitHub repository monitoring** with commit analysis
- **AI-powered content relevance scoring**
- **Automatic categorization** and summarization
- **Multiple post variations** (news commentary, personal experience, community discussion)

### 📅 Smart Scheduling & Publishing
- **Optimal timing analysis** based on audience engagement patterns
- **Priority-based post selection** with quality scoring
- **Automatic LinkedIn publishing** with media support
- **Backup content system** for consistent posting
- **Performance tracking** and learning

### 🖼️ Media Integration
- **Automatic image extraction** from GitHub README files
- **Repository screenshot detection**
- **LinkedIn media upload** with proper formatting
- **Fallback to text-only** if media fails

### 📊 Analytics & Learning
- **Engagement prediction** based on post characteristics
- **Performance analysis** with actionable insights
- **Content strategy optimization**
- **Historical data tracking** in Notion

## 🏗️ Architecture

The system consists of two main pipelines:

### 1. Weekly Content Generation Pipeline (`main.py`)
```
News Scraping → Content Extraction → Relevance Analysis → 
Categorization → Summarization → GitHub Analysis → 
Content Strategy → Post Generation → Review → Scheduling
```

### 2. Daily Publishing Pipeline (`daily_publisher.py`)
```
Fetch Scheduled Posts → Select Best Post → Choose Variation → 
Upload Media → Publish to LinkedIn → Track Performance → Learn & Optimize
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- UV package manager (recommended) or pip
- Active accounts for: LinkedIn, Notion, GitHub, FireCrawl, DeepSeek

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd linkedin-content-manager
   ```

2. **Install dependencies**
   ```bash
   # Using UV (recommended)
   uv sync
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (see Configuration section)
   ```

4. **Test the installation**
   ```bash
   uv run main.py  # Test weekly pipeline
   python daily_publisher.py  # Test daily pipeline
   ```

## ⚙️ Configuration

### Required Environment Variables

Create a `.env` file with the following variables:

```env
# AI/LLM Services
DEEPSEEK_API_KEY=your_deepseek_api_key
OPENROUTER_API_KEY=your_openrouter_key  # Optional alternative

# Content Sources
FIRECRAWL_API_KEY=your_firecrawl_api_key
GITHUB_TOKEN=your_github_personal_access_token
GITHUB_USERNAME=your_github_username

# LinkedIn API
LINKEDIN_ACCESS_TOKEN=your_linkedin_access_token
LINKEDIN_USER_ID=your_linkedin_user_id
LINKEDIN_CLIENT_ID=your_linkedin_client_id

# Notion Integration
NOTION_TOKEN=your_notion_integration_token
NEWS_ARTICLE_DATABASE_ID=your_news_database_id
LINKEDIN_POSTS_DATABASE_ID=your_posts_database_id
GITHUB_DATABASE_ID=your_github_database_id
```

### API Setup Instructions

#### 1. LinkedIn API Setup
1. Create a LinkedIn App at [LinkedIn Developer Portal](https://developer.linkedin.com/)
2. Request these permissions:
   - `r_liteprofile` (read profile)
   - `w_member_social` (post content)
3. Generate access token and get your user ID

#### 2. Notion Setup
1. Create a [Notion Integration](https://www.notion.so/my-integrations)
2. Create three databases with these properties:

**News Articles Database:**
- Title (Title)
- URL (URL)
- Source (Rich Text)
- Relevance Score (Number)
- Category (Select)
- Publication Date (Date)

**LinkedIn Posts Database:**
- Post Title (Title)
- Post Content (Rich Text)
- Post Status (Status: Draft, Scheduled, Published)
- Scheduled Date (Date)
- Scheduled Time (Rich Text)
- Content Quality Score (Number)
- Posting Priority (Select: High, Medium, Low)

**GitHub Database:**
- Repository Name (Title)
- Description (Rich Text)
- Language (Select)
- Stars (Number)
- Last Updated (Date)

#### 3. GitHub Token
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate token with `repo` and `user` permissions

#### 4. FireCrawl API
1. Sign up at [FireCrawl](https://firecrawl.dev/)
2. Get your API key from the dashboard

#### 5. DeepSeek API
1. Sign up at [DeepSeek](https://platform.deepseek.com/)
2. Generate API key for content analysis

## 🚀 Usage

### Weekly Content Generation
Run the main pipeline to generate a week's worth of content:

```bash
uv run main.py
```

This will:
- Scrape latest tech news
- Analyze your GitHub activity
- Generate LinkedIn posts
- Save everything to Notion for review

### Daily Publishing
Run the daily publisher to automatically post scheduled content:

```bash
python daily_publisher.py
```

Or set up automated scheduling:
```bash
# Add to crontab for automatic daily posting
0 9,13,17 * * * cd /path/to/project && python daily_publisher.py
```

### Manual Operations

**Test specific components:**
```bash
# Test news scraping only
python -c "from main import scrape_content; print(scrape_content({}))"

# Test GitHub monitoring
python -c "from main import github_repo_monitor; print(github_repo_monitor())"

# Test LinkedIn publishing
python -c "from daily_publisher import run_daily_publishing; run_daily_publishing()"
```

## 🔌 API Integrations

### Supported News Sources
- **TLDR Tech** - Daily tech newsletter summaries
- **TechCrunch** - Latest startup and tech news
- **Hacker News** - Community-driven tech discussions
- *Easily extensible to add more sources*

### GitHub Integration
- Monitors your repositories for recent commits
- Analyzes code changes and project updates
- Extracts images from README files
- Tracks repository metrics (stars, forks, language)

### LinkedIn Features
- Text and image post publishing
- Optimal timing recommendations
- Engagement prediction
- Performance tracking
- Multiple post variations

### Notion as CMS
- Content review and approval workflow
- Scheduling and calendar management
- Performance analytics storage
- Content strategy planning

## 📁 File Structure

```
linkedin-content-manager/
├── main.py                 # Weekly content generation pipeline
├── daily_publisher.py      # Daily publishing and performance tracking
├── repo.py                 # GitHub repository analysis functions
├── prompt.py               # AI prompt templates for content generation
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create from .env.example)
├── .env.example           # Template for environment variables
├── logs.txt               # Application logs
└── README.md              # This file
```

### Key Components

- **`main.py`**: Orchestrates the weekly content generation workflow
- **`daily_publisher.py`**: Handles daily posting, scheduling, and performance tracking
- **`repo.py`**: GitHub repository monitoring and analysis
- **`prompt.py`**: AI prompt templates for different content generation tasks

## 🐛 Troubleshooting

### Common Issues

#### 1. LinkedIn API Errors
```
Error: LinkedIn API error: 401 - Unauthorized
```
**Solution**: Check your `LINKEDIN_ACCESS_TOKEN` is valid and has required permissions.

#### 2. Notion Database Errors
```
Error: notion_client.errors.APIResponseError: Could not find database
```
**Solution**: Verify database IDs in `.env` and ensure your Notion integration has access.

#### 3. GitHub Rate Limiting
```
Error: RateLimitExceededException
```
**Solution**: The system handles this automatically, but you can increase delays in `repo.py`.

#### 4. Image Upload Failures
```
Warning: Failed to upload image to LinkedIn
```
**Solution**: Check image URLs are publicly accessible and under 5MB.

### Debug Mode
Enable detailed logging by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Testing API Connections
```bash
# Test LinkedIn API
curl -H "Authorization: Bearer YOUR_TOKEN" \
     "https://api.linkedin.com/v2/people/~"

# Test Notion API
curl -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Notion-Version: 2022-06-28" \
     "https://api.notion.com/v1/users/me"
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Adding New News Sources
1. Add source configuration to `news_sources` in `main.py`
2. Update the `extract_tech_news` function if needed
3. Test with the new source

### Extending AI Prompts
1. Add new prompt functions to `prompt.py`
2. Follow the existing pattern for JSON output
3. Update the corresponding pipeline functions

## 📊 Performance Metrics

The system tracks:
- **Content Generation**: Articles processed, posts created, quality scores
- **Publishing Success**: Posts published, failures, timing accuracy
- **Engagement**: Likes, comments, shares, engagement rates
- **Learning**: Performance improvements over time

## 🔒 Security Notes

- Store all API keys in `.env` file (never commit to git)
- Use environment-specific tokens for development/production
- Regularly rotate API keys
- Monitor API usage and rate limits
- Review Notion database permissions

## 📈 Roadmap

- [ ] **Multi-platform support** (Twitter, Medium)
- [ ] **Advanced analytics dashboard**
- [ ] **A/B testing for post variations**
- [ ] **Sentiment analysis integration**
- [ ] **Custom AI model fine-tuning**
- [ ] **Team collaboration features**
- [ ] **Mobile app for content approval**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DeepSeek** for AI-powered content analysis
- **FireCrawl** for reliable web scraping
- **Notion** for content management
- **LinkedIn API** for publishing capabilities
- **GitHub API** for repository monitoring

---

**Built with ❤️ for developers who want to maintain an active LinkedIn presence without the manual effort.**

For questions, issues, or feature requests, please open an issue on GitHub.