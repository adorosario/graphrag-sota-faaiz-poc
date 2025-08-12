#!/usr/bin/env python3
"""
GraphRAG Demo UI - Streamlit Interface
Professional demo for client presentation
"""

import streamlit as st
import asyncio
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import hashlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our backend systems
try:
    from query_router_system import HybridRetriever, RouterConfig
except ImportError:
    st.error(
        "Could not import query_router_system. Please ensure all dependencies are installed.")
    st.stop()

# Load credentials from environment variables
def load_demo_credentials():
    """Load demo credentials from environment variables"""
    credentials = {}
    
    # Load all demo credentials from environment
    for i in range(1, 10):  # Support up to 9 users
        username_key = f"DEMO_USERNAME_{i}"
        password_key = f"DEMO_PASSWORD_{i}"
        
        username = os.getenv(username_key)
        password = os.getenv(password_key)
        
        if username and password:
            credentials[username] = password
    
    # Fallback to default if no env variables found
    if not credentials:
        st.warning("⚠️ No demo credentials found in environment variables. Using defaults.")
        credentials = {
            "demo": "graphrag2025",
            "admin": "admin123", 
            "client": "demo123"
        }
    
    return credentials

DEMO_CREDENTIALS = load_demo_credentials()


def check_authentication():
    """Simple authentication check using environment credentials."""

    # Check if user is already authenticated
    if st.session_state.get('authenticated', False):
        return True

    # Show login form
    st.markdown("""
    <div class="main-header">
        <h1> GraphRAG Assistant </h1>
        <p>Experience the next generation of AI-powered document analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Login form
    with st.form("login_form"):
        st.markdown("### 🔑 Login")
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input(
            "Password", type="password", placeholder="Enter password")
        submitted = st.form_submit_button("🚀 Login")

        if submitted:
            if username in DEMO_CREDENTIALS and DEMO_CREDENTIALS[username] == password:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("✅ Authentication successful! Redirecting...")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Please try again.")

    st.markdown("---")
    st.markdown("""
    **Demo Access:** Contact administrator for credentials
    
    **Environment Status:**
    """)
    
    # Show environment status
    env_status = {
        "Neo4j Connection": "✅ Configured" if os.getenv("NEO4J_URI") else "❌ Missing",
        "Demo Credentials": f"✅ {len(DEMO_CREDENTIALS)} users configured",
        "OpenAI API": "✅ Configured" if os.getenv("OPENAI_API_KEY") else "⚠️ Optional - Not set"
    }
    
    for item, status in env_status.items():
        st.caption(f"{item}: {status}")

    return False


# Page configuration
st.set_page_config(
    page_title="GraphRAG Assistant",
    page_icon="⛁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header { 
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .route-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
        margin: 0.25rem;
        display: inline-block;
    }
    
    .vector-badge { background: #e3f2fd; color: #1565c0; }
    .graph-badge { background: #f3e5f5; color: #7b1fa2; }
    .hybrid-badge { background: #e8f5e8; color: #2e7d32; }
    
    .query-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .result-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
        color: #333333;
    }
    
    .result-header {
        background: #f8f9fa;
        padding: 0.75rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        border-left: 4px solid #667eea;
        color: #333333 !important;
    }
    
    .result-header h5 {
        color: #333333 !important;
        margin: 0;
        font-weight: 600;
    }
    
    .result-header p {
        color: #555555 !important;
        margin: 0.25rem 0;
        font-size: 0.9rem;
    }
    
    .result-header strong {
        color: #333333 !important;
    }
    
    .content-preview {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        color: #444444;
        border: 1px solid #e0e0e0;
        max-height: 200px;
        overflow-y: auto;
    }
    
    .stAlert > div {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False

    if 'hybrid_retriever' not in st.session_state:
        st.session_state.hybrid_retriever = None


@st.cache_resource
def initialize_system():
    """Initialize the GraphRAG system (cached for performance)."""
    try:
        config = RouterConfig()
        hybrid_retriever = HybridRetriever(config)
        return hybrid_retriever
    except Exception as e:
        st.error(f"Failed to initialize GraphRAG system: {str(e)}")
        st.error("Please check your .env file configuration")
        return None


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>GraphRAG Assistant </h1>
        <p>Experience the next generation of AI-powered document analysis.</p>
        <p><strong>Our GraphRAG system combines graph-based knowledge representation with advanced retrieval techniques for unprecedented insights</strong></p>
    </div>
    """, unsafe_allow_html=True)


def render_route_badge(route: str, confidence: float) -> str:
    """Render a styled route badge."""
    badge_class = f"{route.lower()}-badge"
    return f'<span class="route-badge {badge_class}">{route.upper()} ({confidence:.0%})</span>'


async def execute_query(query: str, hybrid_retriever):
    """Execute query and return results."""
    try:
        # Use asyncio to run the query
        response = await hybrid_retriever.retrieve(query)
        return response
    except Exception as e:
        st.error(f"Query execution failed: {str(e)}")
        return None


def display_query_results(response, show_details=True):
    """Display query results with routing decision."""

    if not response:
        st.error("No response received")
        return

    route_decision = response.route_decision

    # Routing Decision Section
    st.markdown("### ⚡ Routing Decision")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        route_badge = render_route_badge(
            route_decision.route.value, route_decision.confidence)
        st.markdown(f"**Route:** {route_badge}", unsafe_allow_html=True)
        st.markdown(f"**Reasoning:** {route_decision.reasoning}")

    with col2:
        st.metric("Confidence", f"{route_decision.confidence:.0%}")
        st.metric("Processing Time", f"{route_decision.processing_time:.3f}s")

    with col3:
        st.metric("Results Found", len(response.results))
        if response.results:
            avg_score = sum(r.score for r in response.results) / \
                len(response.results)
            st.metric("Avg Score", f"{avg_score:.3f}")

    # Query Analysis Details
    if show_details:
        with st.expander("🔧 Query Analysis Details", expanded=False):
            features = route_decision.features

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Query Features:**")
                st.json({
                    "entity_count": features.entity_count,
                    "token_count": features.token_count,
                    "question_type": features.question_type,
                    "complexity_score": round(features.complexity_score, 3)
                })

            with col2:
                st.markdown("**Pattern Detection:**")
                st.json({
                    "multi_hop_indicators": features.has_multi_hop_indicators,
                    "relationship_words": features.has_relationship_words,
                    "complex_reasoning": features.has_complex_reasoning,
                    "named_entities": features.named_entities[:5]
                })

    # Results Section
    st.markdown("### 📄 Retrieved Results")

    if response.results:
        for i, result in enumerate(response.results[:5]):  # Show top 5
            # Use better styling for result cards
            st.markdown(f"""
            <div class="result-header">
                <h5>📄 Result {i+1} (Relevance Score: {result.score:.3f})</h5>
                <p><strong>📂 Source:</strong> {result.source.split('/')[-1]}</p>
                <p><strong>🔀 Route:</strong> {result.route_used.value.upper()}</p>
                {f'<p><strong>🔗 Fusion Sources:</strong> {", ".join(result.metadata.get("fusion_sources", []))}</p>' if result.metadata and "fusion_sources" in result.metadata else ''}
            </div>
            """, unsafe_allow_html=True)

            # Show content in a styled container
            with st.expander(f"📖 View Content - Result {i+1}", expanded=False):
                content_preview = result.content[:800] + "..." if len(
                    result.content) > 800 else result.content
                st.markdown(f"""
                <div class="content-preview">
                    {content_preview}
                </div>
                """, unsafe_allow_html=True)

                # Show metadata if available
                if result.metadata:
                    st.markdown("**📊 Metadata:**")
                    metadata_display = {k: v for k, v in result.metadata.items()
                                        if k not in ['fusion_sources'] and v is not None}
                    if metadata_display:
                        st.json(metadata_display)
    else:
        st.warning(
            "🔍 No results found for this query. Try rephrasing or using different keywords.")


def render_sidebar():
    """Render the sidebar with system info and controls."""
    st.sidebar.markdown("## 🛠 System Status")

    # System Status
    if st.session_state.system_initialized:
        st.sidebar.success("✅ System Ready")
    else:
        st.sidebar.warning("⚠️ Initializing System...")

    # Environment Status
    st.sidebar.markdown("### 🔧 Environment")
    env_checks = {
        "Neo4j": "✅" if os.getenv("NEO4J_URI") else "❌",
        "Credentials": f"✅ ({len(DEMO_CREDENTIALS)} users)",
        "Data Dir": "✅" if os.path.exists(os.getenv("DATA_DIR", "./data")) else "⚠️"
    }
    
    for check, status in env_checks.items():
        st.sidebar.caption(f"{check}: {status}")

    # Document Upload Section
    st.sidebar.markdown("## 📁 Document Upload")

    uploaded_files = st.sidebar.file_uploader(
        "Upload documents to expand knowledge base",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'csv', 'docx', 'md'],
        help="Upload PDF, TXT, CSV, DOCX, or MD files"
    )

    if uploaded_files:
        if st.sidebar.button("🔄 Process Uploaded Documents"):
            with st.spinner("Processing documents..."):
                success_count = process_uploaded_documents(uploaded_files)
                if success_count > 0:
                    st.sidebar.success(
                        f"✅ Processed {success_count} documents!")
                    st.sidebar.info(
                        "System will use new documents in queries.")
                else:
                    st.sidebar.error("❌ Failed to process documents")

    # System Information
    st.sidebar.markdown("### ℹ️ System Info")
    st.sidebar.markdown("""
    **GraphRAG Components:**
    - 🔍 Vector Search (ChromaDB)
    - 🕸 Graph Traversal (Neo4j)
    - 🤖 Intelligent Router
    - ⚡ Hybrid Fusion (RRF)
    """)

    # Query Statistics
    if st.session_state.query_history:
        st.sidebar.markdown("### 📈 Session Stats")

        total_queries = len(st.session_state.query_history)
        st.sidebar.metric("Total Queries", total_queries)

        # Route distribution
        routes = [q['route'] for q in st.session_state.query_history]
        route_counts = {route: routes.count(route) for route in set(routes)}

        for route, count in route_counts.items():
            percentage = (count / total_queries) * 100
            st.sidebar.progress(
                percentage/100, text=f"{route.upper()}: {count} ({percentage:.0f}%)")


def process_uploaded_documents(uploaded_files):
    """Process uploaded documents and add to knowledge base."""
    import os
    from pathlib import Path
    import subprocess
    import sys

    success_count = 0

    # Check file sizes
    max_size_mb = 50
    large_files = []

    for file in uploaded_files:
        size_mb = len(file.getvalue()) / (1024 * 1024)
        if size_mb > max_size_mb:
            large_files.append(f"{file.name} ({size_mb:.1f}MB)")

    if large_files:
        st.sidebar.error(
            f"❌ Files too large (>{max_size_mb}MB): {', '.join(large_files)}")
        return 0

    # Use data directory from environment
    data_dir = os.getenv("DATA_DIR", "./data")
    uploads_dir = Path(data_dir) / "uploads"
    
    try:
        uploads_dir.mkdir(parents=True, exist_ok=True)
        st.sidebar.info(f"📁 Using uploads directory: {uploads_dir}")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to create directory: {e}")
        return 0

    # Save uploaded files
    saved_files = []
    try:
        for uploaded_file in uploaded_files:
            file_path = uploads_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append(str(file_path))
            st.sidebar.success(f"📁 Saved: {uploaded_file.name}")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to save files: {e}")
        return 0

    # Process documents
    try:
        st.sidebar.info("🔄 Processing documents...")

        result = subprocess.run([
            sys.executable, "runtime_processor.py", str(uploads_dir)
        ], capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            success_count = len(uploaded_files)
            st.sidebar.success("✅ Documents processed successfully!")

            # Force reinitialize ChromaDB
            if st.session_state.hybrid_retriever:
                st.sidebar.info("🔄 Updating search index...")
                try:
                    st.session_state.hybrid_retriever.vector_retriever.collection.delete()
                    st.sidebar.info("🗑️ Cleared old index")
                except Exception as e:
                    st.sidebar.warning(f"Index clear warning: {e}")

                try:
                    st.session_state.hybrid_retriever.vector_retriever._ensure_documents_indexed()
                    st.sidebar.success("✅ Search index updated!")
                except Exception as e:
                    st.sidebar.error(f"❌ Index update failed: {e}")

            st.sidebar.info("🎯 New documents ready for queries!")

        else:
            st.sidebar.error(f"❌ Processing failed!")
            st.sidebar.error(f"Error output: {result.stderr}")

    except subprocess.TimeoutExpired:
        st.sidebar.error("❌ Processing timeout (>5 minutes)")
    except Exception as e:
        st.sidebar.error(f"❌ Processing error: {str(e)}")

    return success_count


def render_performance_charts():
    """Render performance charts if we have query history."""

    if not st.session_state.query_history:
        st.info("Execute some queries to see performance charts")
        return

    st.markdown("### 📊 Performance Analytics")

    col1, col2 = st.columns(2)

    with col1:
        # Route distribution pie chart
        routes = [q['route'] for q in st.session_state.query_history]
        route_counts = pd.Series(routes).value_counts()

        fig = px.pie(
            values=route_counts.values,
            names=route_counts.index,
            title="Query Route Distribution",
            color_discrete_map={
                'vector': '#1565c0',
                'graph': '#7b1fa2',
                'hybrid': '#2e7d32'
            }
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Processing time trend
        if len(st.session_state.query_history) > 1:
            df = pd.DataFrame(st.session_state.query_history)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(df))),
                y=df['processing_time'],
                mode='lines+markers',
                name='Processing Time',
                line=dict(color='#667eea')
            ))
            fig.update_layout(
                title="Query Processing Time Trend",
                xaxis_title="Query Number",
                yaxis_title="Time (seconds)",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit application."""

    # Check authentication first
    if not check_authentication():
        return

    # Initialize
    initialize_session_state()

    # Show welcome message with username
    st.sidebar.markdown(
        f"### 👋 Welcome, {st.session_state.get('username', 'User')}!")

    # Logout button
    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()

    # Render header
    render_header()

    # Render sidebar
    render_sidebar()

    # Initialize system if not already done
    if not st.session_state.system_initialized:
        with st.spinner("Initializing GraphRAG system..."):
            hybrid_retriever = initialize_system()
            if hybrid_retriever:
                st.session_state.hybrid_retriever = hybrid_retriever
                st.session_state.system_initialized = True
                st.success("✅ GraphRAG system initialized successfully!")
                st.rerun()
            else:
                st.error(
                    "❌ Failed to initialize system. Please check your .env configuration.")
                return

    # Main interface
    st.markdown("### 🔍 Query Interface")

    # Simplified interface - Only custom query
    selected_query = st.text_area(
        "Enter your query:",
        height=100,
        placeholder="Ask anything about the processed documents..."
    )

    # Execute query
    if selected_query and st.button("🚀 Execute Query", type="primary"):
        with st.spinner("Processing query..."):
            try:
                # Execute the query
                response = asyncio.run(execute_query(
                    selected_query, st.session_state.hybrid_retriever))

                if response:
                    # Display results
                    display_query_results(response)

                    # Add to history
                    st.session_state.query_history.append({
                        'query': selected_query,
                        'route': response.route_decision.route.value,
                        'confidence': response.route_decision.confidence,
                        'processing_time': response.total_processing_time,
                        'results_count': len(response.results),
                        'timestamp': datetime.now()
                    })

            except Exception as e:
                st.error(f"Query execution failed: {str(e)}")

    # Performance charts
    if st.session_state.query_history:
        st.markdown("---")
        render_performance_charts()

    # Query history
    if st.session_state.query_history:
        st.markdown("---")
        st.markdown("### 📝 Query History")

        # Create DataFrame for display
        history_data = []
        for entry in st.session_state.query_history[-10:]:  # Show last 10
            history_data.append({
                'Query': entry['query'][:60] + '...' if len(entry['query']) > 60 else entry['query'],
                'Route': entry['route'].upper(),
                'Confidence': f"{entry['confidence']:.0%}",
                'Time': f"{entry['processing_time']:.3f}s",
                'Results': entry['results_count'],
                'Timestamp': entry['timestamp'].strftime('%H:%M:%S')
            })

        if history_data:
            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
# #!/usr/bin/env python3
# """
# GraphRAG Demo UI - Streamlit Interface
# Professional demo for client presentation
# """

# import streamlit as st
# import asyncio
# import time
# import json
# from datetime import datetime
# from typing import Dict, List, Any
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import hashlib

# # Import our backend systems
# try:
#     from query_router_system import HybridRetriever, RouterConfig
# except ImportError:
#     st.error(
#         "Could not import query_router_system. Please ensure all dependencies are installed.")
#     st.stop()

# # Simple authentication (you can enhance this)
# DEMO_CREDENTIALS = {
#     "demo": "graphrag2025",
#     "admin": "admin123",
#     "client": "demo123"
# }


# def check_authentication():
#     """Simple authentication check."""

#     # Check if user is already authenticated
#     if st.session_state.get('authenticated', False):
#         return True

#     # Show login form
#     st.markdown("""
#     <div class="main-header">
#         <h1> GraphRAG Assistant </h1>
#         <p>Experience the next generation of AI-powered document analysis</p>
#     </div>
#     """, unsafe_allow_html=True)

#     # Login form
#     with st.form("login_form"):
#         st.markdown("### 🔑 Login")
#         username = st.text_input("Username", placeholder="Enter username")
#         password = st.text_input(
#             "Password", type="password", placeholder="Enter password")
#         submitted = st.form_submit_button("🚀 Login")

#         if submitted:
#             if username in DEMO_CREDENTIALS and DEMO_CREDENTIALS[username] == password:
#                 st.session_state.authenticated = True
#                 st.session_state.username = username
#                 st.success("✅ Authentication successful! Redirecting...")
#                 time.sleep(1)
#                 st.rerun()
#             else:
#                 st.error("❌ Invalid credentials. Please try again.")

#     st.markdown("---")
#     st.markdown(
#         "**Contact:** For demo access credentials, please contact the administrator.")

#     return False


# # Page configuration
# st.set_page_config(
#     page_title="GraphRAG Assistant",
#     page_icon="⛁",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for professional styling
# st.markdown("""
# <style>
#     .main-header { 
#         background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
#         padding: 1.5rem;
#         border-radius: 10px;
#         color: white;
#         margin-bottom: 2rem;
#         text-align: center;
#     }
    
#     .metric-card {
#         background: white;
#         padding: 1rem;
#         border-radius: 8px;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#         border-left: 4px solid #667eea;
#     }
    
#     .route-badge {
#         padding: 0.25rem 0.75rem;
#         border-radius: 20px;
#         font-weight: bold;
#         font-size: 0.9rem;
#         margin: 0.25rem;
#         display: inline-block;
#     }
    
#     .vector-badge { background: #e3f2fd; color: #1565c0; }
#     .graph-badge { background: #f3e5f5; color: #7b1fa2; }
#     .hybrid-badge { background: #e8f5e8; color: #2e7d32; }
    
#     .query-container {
#         background: #f8f9fa;
#         padding: 1.5rem;
#         border-radius: 8px;
#         margin: 1rem 0;
#     }
    
#     .result-card {
#         background: white;
#         padding: 1rem;
#         border-radius: 8px;
#         border: 1px solid #e0e0e0;
#         margin: 0.5rem 0;
#         color: #333333;  /* Ensure text is visible */
#     }
    
#     .result-header {
#         background: #f8f9fa;
#         padding: 0.75rem;
#         border-radius: 6px;
#         margin-bottom: 0.5rem;
#         border-left: 4px solid #667eea;
#         color: #333333 !important;  /* Force text color */
#     }
    
#     .result-header h5 {
#         color: #333333 !important;
#         margin: 0;
#         font-weight: 600;
#     }
    
#     .result-header p {
#         color: #555555 !important;
#         margin: 0.25rem 0;
#         font-size: 0.9rem;
#     }
    
#     .result-header strong {
#         color: #333333 !important;
#     }
    
#     .content-preview {
#         background: #f8f9fa;
#         padding: 1rem;
#         border-radius: 6px;
#         font-family: 'Courier New', monospace;
#         font-size: 0.9rem;
#         color: #444444;
#         border: 1px solid #e0e0e0;
#         max-height: 200px;
#         overflow-y: auto;
#     }
    
#     .stAlert > div {
#         padding: 1rem;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize session state


# def initialize_session_state():
#     """Initialize Streamlit session state variables."""
#     if 'query_history' not in st.session_state:
#         st.session_state.query_history = []

#     if 'system_initialized' not in st.session_state:
#         st.session_state.system_initialized = False

#     if 'hybrid_retriever' not in st.session_state:
#         st.session_state.hybrid_retriever = None


# @st.cache_resource
# def initialize_system():
#     """Initialize the GraphRAG system (cached for performance)."""
#     try:
#         config = RouterConfig()
#         hybrid_retriever = HybridRetriever(config)
#         return hybrid_retriever
#     except Exception as e:
#         st.error(f"Failed to initialize GraphRAG system: {str(e)}")
#         return None


# def render_header():
#     """Render the main header."""
#     st.markdown("""
#     <div class="main-header">
#         <h1>GraphRAG Assistant </h1>
#         <p>Experience the next generation of AI-powered document analysis.</p>
#         <p><strong>Our GraphRAG system combines graph-based knowledge representation with advanced retrieval techniques for unprecedented insights</strong></p>
#     </div>
#     """, unsafe_allow_html=True)


# def render_route_badge(route: str, confidence: float) -> str:
#     """Render a styled route badge."""
#     badge_class = f"{route.lower()}-badge"
#     return f'<span class="route-badge {badge_class}">{route.upper()} ({confidence:.0%})</span>'


# async def execute_query(query: str, hybrid_retriever):
#     """Execute query and return results."""
#     try:
#         # Use asyncio to run the query
#         response = await hybrid_retriever.retrieve(query)
#         return response
#     except Exception as e:
#         st.error(f"Query execution failed: {str(e)}")
#         return None


# def display_query_results(response, show_details=True):
#     """Display query results with routing decision."""

#     if not response:
#         st.error("No response received")
#         return

#     route_decision = response.route_decision

#     # Routing Decision Section
#     st.markdown("### ⚡ Routing Decision")

#     col1, col2, col3 = st.columns([2, 1, 1])

#     with col1:
#         route_badge = render_route_badge(
#             route_decision.route.value, route_decision.confidence)
#         st.markdown(f"**Route:** {route_badge}", unsafe_allow_html=True)
#         st.markdown(f"**Reasoning:** {route_decision.reasoning}")

#     with col2:
#         st.metric("Confidence", f"{route_decision.confidence:.0%}")
#         st.metric("Processing Time", f"{route_decision.processing_time:.3f}s")

#     with col3:
#         st.metric("Results Found", len(response.results))
#         if response.results:
#             avg_score = sum(r.score for r in response.results) / \
#                 len(response.results)
#             st.metric("Avg Score", f"{avg_score:.3f}")

#     # Query Analysis Details
#     if show_details:
#         with st.expander("🔧 Query Analysis Details", expanded=False):
#             features = route_decision.features

#             col1, col2 = st.columns(2)

#             with col1:
#                 st.markdown("**Query Features:**")
#                 st.json({
#                     "entity_count": features.entity_count,
#                     "token_count": features.token_count,
#                     "question_type": features.question_type,
#                     "complexity_score": round(features.complexity_score, 3)
#                 })

#             with col2:
#                 st.markdown("**Pattern Detection:**")
#                 st.json({
#                     "multi_hop_indicators": features.has_multi_hop_indicators,
#                     "relationship_words": features.has_relationship_words,
#                     "complex_reasoning": features.has_complex_reasoning,
#                     # Show first 5
#                     "named_entities": features.named_entities[:5]
#                 })

#     # Results Section
#     st.markdown("### 📄 Retrieved Results")

#     if response.results:
#         for i, result in enumerate(response.results[:5]):  # Show top 5
#             # Use better styling for result cards
#             st.markdown(f"""
#             <div class="result-header">
#                 <h5>📄 Result {i+1} (Relevance Score: {result.score:.3f})</h5>
#                 <p><strong>📂 Source:</strong> {result.source.split('/')[-1]}</p>
#                 <p><strong>🔀 Route:</strong> {result.route_used.value.upper()}</p>
#                 {f'<p><strong>🔗 Fusion Sources:</strong> {", ".join(result.metadata.get("fusion_sources", []))}</p>' if result.metadata and "fusion_sources" in result.metadata else ''}
#             </div>
#             """, unsafe_allow_html=True)

#             # Show content in a styled container
#             with st.expander(f"📖 View Content - Result {i+1}", expanded=False):
#                 content_preview = result.content[:800] + "..." if len(
#                     result.content) > 800 else result.content
#                 st.markdown(f"""
#                 <div class="content-preview">
#                     {content_preview}
#                 </div>
#                 """, unsafe_allow_html=True)

#                 # Show metadata if available
#                 if result.metadata:
#                     st.markdown("**📊 Metadata:**")
#                     metadata_display = {k: v for k, v in result.metadata.items()
#                                         if k not in ['fusion_sources'] and v is not None}
#                     if metadata_display:
#                         st.json(metadata_display)
#     else:
#         st.warning(
#             "🔍 No results found for this query. Try rephrasing or using different keywords.")


# def render_sidebar():
#     """Render the sidebar with system info and controls."""
#     st.sidebar.markdown("## 🛠 System Status")

#     # System Status
#     if st.session_state.system_initialized:
#         st.sidebar.success("✅ System Ready")
#     else:
#         st.sidebar.warning("⚠️ Initializing System...")

#     # Document Upload Section
#     st.sidebar.markdown("## 📁 Document Upload")

#     uploaded_files = st.sidebar.file_uploader(
#         "Upload documents to expand knowledge base",
#         accept_multiple_files=True,
#         type=['pdf', 'txt', 'csv', 'docx', 'md'],
#         help="Upload PDF, TXT, CSV, DOCX, or MD files"
#     )

#     if uploaded_files:
#         if st.sidebar.button("🔄 Process Uploaded Documents"):
#             # Use st.spinner instead of st.sidebar.spinner
#             with st.spinner("Processing documents..."):
#                 success_count = process_uploaded_documents(uploaded_files)
#                 if success_count > 0:
#                     st.sidebar.success(
#                         f"✅ Processed {success_count} documents!")
#                     st.sidebar.info(
#                         "System will use new documents in queries.")
#                 else:
#                     st.sidebar.error("❌ Failed to process documents")

#     # System Information
#     st.sidebar.markdown("### ℹ️ System Info")
#     st.sidebar.markdown("""
#     **GraphRAG Components:**
#     - 🔍 Vector Search (ChromaDB)
#     - 🕸 Graph Traversal (Neo4j)
#     - 🤖 Intelligent Router
#     - ⚡ Hybrid Fusion (RRF)
#     """)

#     # Query Statistics
#     if st.session_state.query_history:
#         st.sidebar.markdown("### 📈 Session Stats")

#         total_queries = len(st.session_state.query_history)
#         st.sidebar.metric("Total Queries", total_queries)

#         # Route distribution
#         routes = [q['route'] for q in st.session_state.query_history]
#         route_counts = {route: routes.count(route) for route in set(routes)}

#         for route, count in route_counts.items():
#             percentage = (count / total_queries) * 100
#             st.sidebar.progress(
#                 percentage/100, text=f"{route.upper()}: {count} ({percentage:.0f}%)")


# def process_uploaded_documents(uploaded_files):
#     """Process uploaded documents and add to knowledge base."""
#     import os
#     from pathlib import Path
#     import subprocess
#     import sys

#     success_count = 0

#     # Check file sizes (Streamlit default limit is 200MB)
#     max_size_mb = 50  # Set reasonable limit
#     large_files = []

#     for file in uploaded_files:
#         size_mb = len(file.getvalue()) / (1024 * 1024)
#         if size_mb > max_size_mb:
#             large_files.append(f"{file.name} ({size_mb:.1f}MB)")

#     if large_files:
#         st.sidebar.error(
#             f"❌ Files too large (>{max_size_mb}MB): {', '.join(large_files)}")
#         return 0

#     # Create uploads directory in the project
#     uploads_dir = Path("/opt/graphrag/graphrag-poc/data/uploads")
#     try:
#         uploads_dir.mkdir(parents=True, exist_ok=True)
#         st.sidebar.info(f"📁 Created uploads directory: {uploads_dir}")
#     except Exception as e:
#         st.sidebar.error(f"❌ Failed to create directory: {e}")
#         return 0

#     # Save uploaded files to permanent location
#     saved_files = []
#     try:
#         for uploaded_file in uploaded_files:
#             file_path = uploads_dir / uploaded_file.name
#             with open(file_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())
#             saved_files.append(str(file_path))
#             st.sidebar.success(f"📁 Saved: {uploaded_file.name}")
#     except Exception as e:
#         st.sidebar.error(f"❌ Failed to save files: {e}")
#         return 0

#     # Process documents using our existing pipeline
#     try:
#         st.sidebar.info("🔄 Processing documents...")

#         # Change to the correct directory and run processor
#         result = subprocess.run([
#             sys.executable, "runtime_processor.py", str(uploads_dir)
#         ], capture_output=True, text=True, cwd="/opt/graphrag/graphrag-poc", timeout=300)

#         if result.returncode == 0:
#             success_count = len(uploaded_files)
#             st.sidebar.success("✅ Documents processed successfully!")

#             # Force reinitialize ChromaDB to include new documents
#             if st.session_state.hybrid_retriever:
#                 st.sidebar.info("🔄 Updating search index...")
#                 try:
#                     # Clear the existing collection
#                     st.session_state.hybrid_retriever.vector_retriever.collection.delete()
#                     st.sidebar.info("🗑️ Cleared old index")
#                 except Exception as e:
#                     st.sidebar.warning(f"Index clear warning: {e}")

#                 # Reinitialize the collection
#                 try:
#                     st.session_state.hybrid_retriever.vector_retriever._ensure_documents_indexed()
#                     st.sidebar.success("✅ Search index updated!")
#                 except Exception as e:
#                     st.sidebar.error(f"❌ Index update failed: {e}")

#             st.sidebar.info("🎯 New documents ready for queries!")

#         else:
#             st.sidebar.error(f"❌ Processing failed!")
#             st.sidebar.error(f"Error output: {result.stderr}")

#     except subprocess.TimeoutExpired:
#         st.sidebar.error("❌ Processing timeout (>5 minutes)")
#     except Exception as e:
#         st.sidebar.error(f"❌ Processing error: {str(e)}")

#     return success_count


# def render_performance_charts():
#     """Render performance charts if we have query history."""

#     if not st.session_state.query_history:
#         st.info("Execute some queries to see performance charts")
#         return

#     st.markdown("### 📊 Performance Analytics")

#     col1, col2 = st.columns(2)

#     with col1:
#         # Route distribution pie chart
#         routes = [q['route'] for q in st.session_state.query_history]
#         route_counts = pd.Series(routes).value_counts()

#         fig = px.pie(
#             values=route_counts.values,
#             names=route_counts.index,
#             title="Query Route Distribution",
#             color_discrete_map={
#                 'vector': '#1565c0',
#                 'graph': '#7b1fa2',
#                 'hybrid': '#2e7d32'
#             }
#         )
#         fig.update_traces(textposition='inside', textinfo='percent+label')
#         st.plotly_chart(fig, use_container_width=True)

#     with col2:
#         # Processing time trend
#         if len(st.session_state.query_history) > 1:
#             df = pd.DataFrame(st.session_state.query_history)

#             fig = go.Figure()
#             fig.add_trace(go.Scatter(
#                 x=list(range(len(df))),
#                 y=df['processing_time'],
#                 mode='lines+markers',
#                 name='Processing Time',
#                 line=dict(color='#667eea')
#             ))
#             fig.update_layout(
#                 title="Query Processing Time Trend",
#                 xaxis_title="Query Number",
#                 yaxis_title="Time (seconds)",
#                 showlegend=False
#             )
#             st.plotly_chart(fig, use_container_width=True)


# def main():
#     """Main Streamlit application."""

#     # Check authentication first
#     if not check_authentication():
#         return

#     # Initialize
#     initialize_session_state()

#     # Show welcome message with username
#     st.sidebar.markdown(
#         f"### 👋 Welcome, {st.session_state.get('username', 'User')}!")

#     # Logout button
#     if st.sidebar.button("🚪 Logout"):
#         st.session_state.authenticated = False
#         st.session_state.username = None
#         st.rerun()

#     # Render header
#     render_header()

#     # Render sidebar
#     render_sidebar()

#     # Initialize system if not already done
#     if not st.session_state.system_initialized:
#         with st.spinner("Initializing GraphRAG system..."):
#             hybrid_retriever = initialize_system()
#             if hybrid_retriever:
#                 st.session_state.hybrid_retriever = hybrid_retriever
#                 st.session_state.system_initialized = True
#                 st.success("✅ GraphRAG system initialized successfully!")
#                 st.rerun()
#             else:
#                 st.error(
#                     "❌ Failed to initialize system. Please check your configuration.")
#                 return

#     # Main interface
#     st.markdown("### 🔍 Query Interface")

#     # Simplified interface - Only custom query (removed Demo Queries)
#     selected_query = st.text_area(
#         "Enter your query:",
#         height=100,
#         placeholder="Ask anything about the processed documents..."
#     )

#     # Execute query
#     if selected_query and st.button("🚀 Execute Query", type="primary"):
#         with st.spinner("Processing query..."):
#             try:
#                 # Execute the query
#                 response = asyncio.run(execute_query(
#                     selected_query, st.session_state.hybrid_retriever))

#                 if response:
#                     # Display results
#                     display_query_results(response)

#                     # Add to history
#                     st.session_state.query_history.append({
#                         'query': selected_query,
#                         'route': response.route_decision.route.value,
#                         'confidence': response.route_decision.confidence,
#                         'processing_time': response.total_processing_time,
#                         'results_count': len(response.results),
#                         'timestamp': datetime.now()
#                     })

#             except Exception as e:
#                 st.error(f"Query execution failed: {str(e)}")

#     # Performance charts
#     if st.session_state.query_history:
#         st.markdown("---")
#         render_performance_charts()

#     # Query history
#     if st.session_state.query_history:
#         st.markdown("---")
#         st.markdown("### 📝 Query History")

#         # Create DataFrame for display
#         history_data = []
#         for entry in st.session_state.query_history[-10:]:  # Show last 10
#             history_data.append({
#                 'Query': entry['query'][:60] + '...' if len(entry['query']) > 60 else entry['query'],
#                 'Route': entry['route'].upper(),
#                 'Confidence': f"{entry['confidence']:.0%}",
#                 'Time': f"{entry['processing_time']:.3f}s",
#                 'Results': entry['results_count'],
#                 'Timestamp': entry['timestamp'].strftime('%H:%M:%S')
#             })

#         if history_data:
#             df = pd.DataFrame(history_data)
#             st.dataframe(df, use_container_width=True, hide_index=True)


# if __name__ == "__main__":
#     main()