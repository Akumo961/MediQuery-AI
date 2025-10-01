import requests


def quick_test():
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/health")
        print(f"✅ Health Check: {response.status_code} - {response.json()}")

        # Test main endpoint
        response = requests.get("http://localhost:8000/")
        print(f"✅ Main API: {response.status_code}")

        # Test search suggestions
        response = requests.get("http://localhost:8000/api/search/suggestions?q=cancer")
        print(f"✅ Search Suggestions: {response.status_code}")

    except Exception as e:
        print(f"❌ API not running: {e}")


if __name__ == "__main__":
    quick_test()