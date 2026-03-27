from newspaper import Article
import json

if __name__ == "__main__":
    # Real article URL from dev.to
    test_url = "https://dev.to/redis/building-reliable-agents-with-the-transactional-outbox-pattern-and-redis-streams-45e6" 
    
    # Let's override the default download() behavior for better reliability
    from newspaper import Article, Config
    
    config = Config()
    config.request_timeout = 20 # Longer timeout for slow networks
    config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    
    print(f"Testing extraction for: {test_url}")
    try:
        article = Article(test_url, config=config)
        article.download()
        article.parse()
        
        results = {
            "title": article.title,
            "top_image": article.top_image,
            "images": list(article.images)[:5],
            "publish_date": str(article.publish_date) if article.publish_date else "None",
            "authors": article.authors
        }
        
        print("\nExtraction Successful!")
        print(json.dumps(results, indent=2))
        
        if not article.top_image:
            print("\nNo 'top_image' found. Checking for all images...")
            if article.images:
                print(f"Found {len(article.images)} images. Top candidate: {list(article.images)[0]}")
            else:
                print("No images found on this page.")
                
    except Exception as e:
        print(f"\nError: {e}")
