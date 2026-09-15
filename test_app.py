import unittest
from fastapi.testclient import TestClient
from app import app, load_data_and_train


class TestCineMatchApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_data_and_train()
        cls.client = TestClient(app)

    def test_serve_index(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('CineMatch', response.text)
        self.assertIn('text/html', response.headers.get('content-type', ''))

    def test_get_movies(self):
        response = self.client.get('/api/movies')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        first = data[0]
        self.assertIn('id', first)
        self.assertIn('title', first)
        self.assertIn('genre', first)

    def test_seed_recommendations_all_strategies(self):
        payload = {'seed_id': 1, 'user_ratings': {}}
        response = self.client.post('/api/recommend', json=payload)
        self.assertEqual(response.status_code, 200)
        res = response.json()
        
        self.assertIn('seed_movie', res)
        self.assertEqual(res['seed_movie']['id'], 1)
        
        seed_based = res['seed_based']
        self.assertIn('content_based', seed_based)
        self.assertIn('collaborative', seed_based)
        self.assertIn('hybrid', seed_based)
        
        self.assertEqual(len(seed_based['content_based']), 5)
        self.assertEqual(len(seed_based['collaborative']), 5)
        self.assertEqual(len(seed_based['hybrid']), 5)
        
        # Check recommendation structure
        top_hybrid = seed_based['hybrid'][0]
        self.assertIn('id', top_hybrid)
        self.assertIn('title', top_hybrid)
        self.assertIn('genre', top_hybrid)
        self.assertIn('score', top_hybrid)
        self.assertNotEqual(top_hybrid['id'], 1)

    def test_personalized_recommendations_all_strategies(self):
        # User rates Sci-Fi high (id 1: 5.0) and Romance low (id 21: 1.0)
        payload = {
            'seed_id': 1,
            'user_ratings': {'1': 5.0, '21': 1.0}
        }
        response = self.client.post('/api/recommend', json=payload)
        self.assertEqual(response.status_code, 200)
        res = response.json()
        
        pers = res['personalized']
        self.assertIn('content_based', pers)
        self.assertIn('collaborative', pers)
        self.assertIn('hybrid', pers)
        
        self.assertGreater(len(pers['content_based']), 0)
        self.assertGreater(len(pers['collaborative']), 0)
        self.assertGreater(len(pers['hybrid']), 0)

    def test_invalid_seed_id(self):
        payload = {'seed_id': 99999, 'user_ratings': {}}
        response = self.client.post('/api/recommend', json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn('not found', response.json().get('detail', ''))


if __name__ == '__main__':
    unittest.main()
