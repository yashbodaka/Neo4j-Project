"""
GMP Regulatory Intelligence - Web Interface
Professional Flask application for querying the RAG system.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from rag.answer_generator import AnswerGenerator

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')

# Initialize RAG system
gemini_api_key = os.getenv('GEMINI_API_KEY')
generator = AnswerGenerator(api_key=gemini_api_key)


@app.route('/')
def index():
    """Render main interface."""
    return render_template('index.html')


@app.route('/api/query', methods=['POST'])
def query():
    """Process RAG query and return results."""
    try:
        data = request.get_json()
        query_text = data.get('query', '').strip()
        top_k = data.get('top_k', 7)
        
        if not query_text:
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty'
            }), 400
        
        # Generate answer
        answer = generator.generate_answer(query_text, top_k=top_k)
        
        # Format response
        return jsonify({
            'success': True,
            'answer': {
                'text': answer.answer,
                'confidence_score': answer.confidence_score,
                'cypher_query': answer.cypher_query,
                'sources': [
                    {
                        'source_doc': s.get('source_doc', ''),
                        'requirement_id': s.get('section_id', ''),
                        'text': s.get('excerpt', ''),
                        'category': s.get('citation_text', '')
                    }
                    for s in answer.sources
                ],
                'graph_path': answer.graph_path,
                'path_description': answer.path_description,
                'conflicts': answer.conflicts,
                'related_requirements': answer.related_requirements
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'GMP Regulatory Intelligence',
        'rag_system': 'operational'
    })


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 GMP REGULATORY INTELLIGENCE - WEB INTERFACE")
    print("="*80)
    print(f"\n📊 RAG System: {generator.reasoner.model_hierarchy}")
    print(f"🔗 Access at: http://localhost:5000")
    print("\n" + "="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
