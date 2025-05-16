import requests

SEARCH_GUTENBERG_BOOKS_TOOL = {
        "type": "function",
        "function": {
            "name": "search_gutenberg_books",
            "description": "Search for books in the Project Gutenberg library.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Search terms for book lookup"
                    }
                },
                "required": ["search_terms"]
            }
        }
    }

def search_gutenberg_books(search_terms):
    resp = requests.get("https://gutendex.com/books", params={"search": " ".join(search_terms)})
    return [{"title": b["title"]} for b in resp.json().get("results", [])[:5]]