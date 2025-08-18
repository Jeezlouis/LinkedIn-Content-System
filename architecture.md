# LinkedIn Automation Architecture

```mermaid
graph TD
    %% Data Sources
    A[Tech News Sources<br/>TechCrunch, HackerNews, Wired] --> B[News Crawler Agent]
    C[GitHub Repositories<br/>Your Projects] --> D[Repository Monitor Agent]
    
    %% LLM-Powered Agents
    B --> E{LLM News Analysis<br/>Relevance & Categorization}
    D --> F{LLM Code Analysis<br/>Significance Detection}
    
    %% Shared State
    E --> G[Shared State<br/>News + Repo Updates<br/>Historical Data<br/>Performance Metrics]
    F --> G
    
    %% Decision Engine
    G --> H{Content Strategist Agent<br/>LLM Decision Making}
    
    %% Content Strategy Paths
    H -->|High Value News| I[News Commentary Path]
    H -->|Significant Repo Update| J[Project Showcase Path]
    H -->|Combined Insights| K[Thought Leadership Path]
    H -->|No Quality Content| L[Wait State]
    
    %% Content Generation
    I --> M[Content Writer Agent<br/>LLM-Powered Writing]
    J --> M
    K --> M
    
    %% Quality Control
    M --> N{Content Review Agent<br/>LLM Quality Check}
    N -->|Needs Revision| M
    N -->|Approved| O[Content Queue]
    
    %% Publishing Pipeline
    O --> P{Scheduler Agent<br/>Optimal Timing}
    P --> Q[LinkedIn API<br/>Auto-Publishing]
    
    %% Feedback Loop
    Q --> R[Engagement Tracking]
    R --> S{Performance Analysis<br/>LLM Learning}
    S --> G
    
    %% Manual Overrides
    T[Manual Review<br/>Emergency Stop] -.-> O
    T -.-> Q
    
    %% LLM Services
    U[LLM Provider<br/>OpenAI/Anthropic/Local] -.-> E
    U -.-> F
    U -.-> H
    U -.-> M
    U -.-> N
    U -.-> S
    
    %% External APIs
    V[Firecrawl API] -.-> B
    W[GitHub API] -.-> D
    X[LinkedIn API] -.-> Q
    
    %% State Storage
    Y[Database/Files<br/>Historical Data<br/>Content Queue<br/>Performance Metrics] -.-> G
    
    %% Styling
    classDef agent fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef llm fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef api fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef decision fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef storage fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class B,D,M,R agent
    class E,F,H,N,S,U llm
    class V,W,X,Q api
    class H,N,P decision
    class G,Y storage


