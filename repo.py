from datetime import datetime, timedelta
import re

def extract_repo_images(repo, readme_content=None):
    """
    Extract various image URLs from a GitHub repository
    
    Args:
        repo: PyGithub Repository object
        readme_content: README content object (optional)
        
    Returns:
        dict: Dictionary containing different types of image URLs
    """
    images = {
        'social_preview': None,      # Repository social preview image
        'owner_avatar': None,        # Repository owner's avatar
        'readme_images': [],         # Images found in README
        'primary_image': None        # Best representative image
    }
    
    try:
        # 1. Repository owner's avatar
        if repo.owner:
            images['owner_avatar'] = repo.owner.avatar_url
        
        # 2. Extract images from README content
        if readme_content:
            try:
                readme_text = readme_content.decoded_content.decode('utf-8')
                images['readme_images'] = extract_images_from_markdown(readme_text, repo)
            except Exception as e:
                print(f"⚠️ Could not parse README images for {repo.name}: {e}")
        
        # 3. Determine primary image (best representative image)
        if images['readme_images']:
            # Use first README image as primary
            images['primary_image'] = images['readme_images'][0]
        elif images['owner_avatar']:
            # Fallback to owner avatar
            images['primary_image'] = images['owner_avatar']
        
        # 4. Try to get repository topics for context
        try:
            topics = repo.get_topics()
            images['topics'] = topics  # Not an image, but useful context
        except:
            images['topics'] = []
            
    except Exception as e:
        print(f"⚠️ Error extracting images from {repo.name}: {e}")
    
    return images

def extract_images_from_markdown(markdown_text, repo):
    """
    Extract image URLs from markdown content
    
    Args:
        markdown_text: README markdown content
        repo: PyGithub Repository object for resolving relative URLs
        
    Returns:
        list: List of image URLs found in the markdown
    """
    image_urls = []
    
    # Regex patterns for different markdown image formats
    patterns = [
        r'!\[.*?\]\((.*?)\)',           # ![alt](url)
        r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>',  # <img src="url">
        r'!\[.*?\]:\s*(.*?)(?:\s|$)',   # ![alt]: url
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, markdown_text, re.IGNORECASE)
        for match in matches:
            url = match.strip()
            
            # Skip empty URLs
            if not url:
                continue
                
            # Convert relative URLs to absolute GitHub URLs
            if url.startswith('./') or url.startswith('../') or not url.startswith('http'):
                # Remove leading ./ or ../
                clean_url = url.lstrip('./')
                # Build absolute GitHub URL
                base_url = f"https://raw.githubusercontent.com/{repo.full_name}/{repo.default_branch}"
                url = f"{base_url}/{clean_url}"
            
            # Filter for image file extensions
            if any(ext in url.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp']):
                if url not in image_urls:  # Avoid duplicates
                    image_urls.append(url)
    
    return image_urls[:5]  # Limit to first 5 images to avoid clutter

def analyze_single_repo(repo):
    """
    Analyze a single PyGithub repository object
    Extracts recent commits, statistics, and metadata for LLM analysis
    
    Args:
        repo: PyGithub Repository object
        
    Returns:
        dict: Structured repository data or None if no recent activity
    """
    
    # Get recent commits (last 7 days)
    since_date = datetime.now() - timedelta(days=7)
    
    try:
        # Get commits as a paginated list first, then convert to list
        commits_paginated = repo.get_commits(since=since_date, author=repo.owner)
        commits = list(commits_paginated)
    except Exception as e:
        print(f"⚠️ Could not get commits for {repo.name}: {e}")
        return None
    
    if not commits:  # No recent activity
        print(f"📭 No recent commits in {repo.name}")
        return None
    
    # Extract commit data
    commit_data = []
    for commit in commits[:20]:  # Last 20 commits max
        try:
            # Handle files count safely - files might be a PaginatedList
            files_count = 0
            if commit.files:
                try:
                    files_count = len(list(commit.files))
                except:
                    files_count = commit.stats.total if commit.stats else 0
            
            commit_data.append({
                'message': commit.commit.message,
                'sha': commit.sha[:7],  # Short hash
                'date': commit.commit.author.date.isoformat(),
                'files_changed': files_count,
                'additions': commit.stats.additions if commit.stats else 0,
                'deletions': commit.stats.deletions if commit.stats else 0,
                'url': commit.html_url
            })
        except Exception as e:
            print(f"⚠️ Error processing commit in {repo.name}: {e}")
            continue
    
    if not commit_data:  # No processable commits
        print(f"❌ No processable commits in {repo.name}")
        return None
    
    # Calculate total changes
    total_additions = sum(c['additions'] for c in commit_data)
    total_deletions = sum(c['deletions'] for c in commit_data)
    total_files = sum(c['files_changed'] for c in commit_data)

    # Get README content safely
    content_summary = None
    try:
        content_summary = repo.get_contents("README.md")
    except Exception as e:
        print(f"⚠️ Could not get README for {repo.name}: {e}")
    
    # Extract image URLs from repository
    image_urls = extract_repo_images(repo, content_summary)
    
    # Prepare structured repository data
    repo_analysis = {
        'name': repo.name,
        'description': repo.description or 'No description available',
        'content_summary': content_summary.decoded_content.decode('utf-8') if content_summary else 'No README available',
        'language': repo.language or 'Unknown',
        'stars': repo.stargazers_count,
        'forks': repo.forks_count,
        'recent_commits': commit_data,
        'commits_count': len(commits),
        'total_additions': total_additions,
        'total_deletions': total_deletions,
        'total_files_changed': total_files,
        'last_push': repo.pushed_at.isoformat() if repo.pushed_at else None,
        'repo_url': repo.html_url,
        'is_private': repo.private,
        'created_at': repo.created_at.isoformat() if repo.created_at else None,
        'default_branch': repo.default_branch,
        'images': image_urls  # Add image URLs
    }
    
    print(f"✅ Analyzed {repo.name}: {len(commits)} commits, +{total_additions}/-{total_deletions} lines")
    return repo_analysis