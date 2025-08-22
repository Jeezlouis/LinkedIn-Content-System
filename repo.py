from datetime import datetime, timedelta

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
    
    # Prepare structured repository data
    repo_analysis = {
        'name': repo.name,
        'description': repo.description or 'No description available',
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
        'default_branch': repo.default_branch
    }
    
    print(f"✅ Analyzed {repo.name}: {len(commits)} commits, +{total_additions}/-{total_deletions} lines")
    return repo_analysis