from flask import Blueprint, jsonify, request
import arxiv

bp = Blueprint('search', __name__, url_prefix='/api')

@bp.route('/search')
def search_papers():
    try:
        search_query = request.args.get('q')
        
        # 使用arxiv API搜索论文
        client = arxiv.Client()
        search = arxiv.Search(
            query=search_query,
            max_results=20,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        papers = []
        for result in client.results(search):
            papers.append({
                'id': result.get_short_id(),
                'title': result.title,
                'authors': ', '.join(a.name for a in result.authors),
                'categories': ', '.join(result.categories),
                'published': result.published.strftime('%Y-%m-%d'),
                'summary': result.summary,
                'url': result.entry_id,
            })
        
        return jsonify(papers)  # 返回前10条结果
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500