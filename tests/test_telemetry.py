import unittest
import os
import json
from bs4 import BeautifulSoup
from app import app
from services.telemetry_service import get_telemetry_data


class TestTelemetryPageAndAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_telemetry_api_default(self):
        """Test default GET /api/telemetry returns status success, summary, and time series."""
        resp = self.client.get('/api/telemetry')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get("status"), "success")
        
        # Check summary keys
        summary = data.get("summary", {})
        for key in ["total_prompts", "total_responses", "total_errors", "total_input_tokens", "total_output_tokens"]:
            self.assertIn(key, summary)
            self.assertIsInstance(summary[key], int)

        # Check models and config
        self.assertIn("models", data)
        self.assertEqual(data.get("selected_model"), "all")
        filters = data.get("filters", {})
        self.assertEqual(filters.get("interval"), "15m")
        self.assertEqual(filters.get("time_range"), "1d")

        # Check time series structure
        series = data.get("time_series", {})
        self.assertIsInstance(series, dict)
        for metric in ["labels", "prompts", "responses", "errors", "input_tokens", "output_tokens"]:
            self.assertIn(metric, series)
            self.assertIsInstance(series[metric], list)
            self.assertGreater(len(series[metric]), 0)

    def test_telemetry_api_model_filter(self):
        """Test GET /api/telemetry with model filter."""
        resp_all = self.client.get('/api/telemetry')
        models = resp_all.get_json().get("models", [])
        if models:
            selected = models[0]
            resp_filtered = self.client.get(f'/api/telemetry?model={selected}')
            self.assertEqual(resp_filtered.status_code, 200)
            data_filtered = resp_filtered.get_json()
            self.assertEqual(data_filtered.get("selected_model"), selected)

    def test_telemetry_api_intervals_and_ranges(self):
        """Test GET /api/telemetry with different intervals and time ranges."""
        for interval in ["1m", "15m", "1h", "1d"]:
            resp = self.client.get(f'/api/telemetry?interval={interval}&time_range=week')
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.get_json().get("filters", {}).get("interval"), interval)

        for tr in ["1h", "1d", "week", "month"]:
            resp = self.client.get(f'/api/telemetry?time_range={tr}')
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.get_json().get("filters", {}).get("time_range"), tr)

    def test_telemetry_api_custom_date_range(self):
        """Test GET /api/telemetry with custom date range."""
        resp = self.client.get('/api/telemetry?time_range=custom&start_date=2026-09-01&end_date=2026-09-15')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get("filters", {}).get("time_range"), "custom")

    def test_html_telemetry_page_elements(self):
        """Verify templates/index.html includes the Telemetry tab, order, controls, plots, and educational card."""
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        soup = BeautifulSoup(resp.data.decode('utf-8'), 'html.parser')

        # 1. Navigation tab ordering: Ingest (2) -> Telemetry (3) -> Audit Logs (4)
        tabs = soup.select('.top-nav-tabs .nav-tab')
        tab_ids = [t.get('id') for t in tabs if t.get('id')]
        self.assertIn('tabBtnIngest', tab_ids)
        self.assertIn('tabBtnTelemetry', tab_ids)
        self.assertIn('tabBtnLogs', tab_ids)
        ingest_idx = tab_ids.index('tabBtnIngest')
        telemetry_idx = tab_ids.index('tabBtnTelemetry')
        logs_idx = tab_ids.index('tabBtnLogs')
        self.assertLess(ingest_idx, telemetry_idx, "Telemetry tab must be after Ingest tab")
        self.assertLess(telemetry_idx, logs_idx, "Telemetry tab must be before Logs tab")

        # 2. Check Top Statistics IDs
        for stat_id in [
            'statTotalPrompts',
            'statTotalResponses',
            'statTotalErrors',
            'statTotalInputTokens',
            'statTotalOutputTokens'
        ]:
            el = soup.find(id=stat_id)
            self.assertIsNotNone(el, f"Missing metric element #{stat_id}")

        # 3. Check Model Filter Dropdown & Refresh Button
        model_select = soup.find('select', id='telemetryModelSelect')
        self.assertIsNotNone(model_select)
        options = [opt.text.strip() for opt in model_select.find_all('option')]
        self.assertIn('All Models', options)
        refresh_btn = soup.find('button', id='btnRefreshTelemetry')
        self.assertIsNotNone(refresh_btn)

        # 4. Check Interval & Time Range Selectors
        interval_select = soup.find('select', id='telemetryIntervalSelect')
        self.assertIsNotNone(interval_select)
        int_options = [opt.get('value') for opt in interval_select.find_all('option')]
        self.assertEqual(int_options, ['1m', '15m', '1h', '1d'])
        default_interval = interval_select.find('option', selected=True)
        self.assertEqual(default_interval.get('value'), '15m')

        time_range_select = soup.find('select', id='telemetryTimeRangeSelect')
        self.assertIsNotNone(time_range_select)
        tr_options = [opt.get('value') for opt in time_range_select.find_all('option')]
        self.assertEqual(tr_options, ['1h', '1d', 'week', 'month', 'custom'])
        default_tr = time_range_select.find('option', selected=True)
        self.assertEqual(default_tr.get('value'), '1d')

        # 5. Check Custom Date Pickers
        start_date = soup.find('input', id='telemetryStartDate')
        end_date = soup.find('input', id='telemetryEndDate')
        self.assertIsNotNone(start_date)
        self.assertIsNotNone(end_date)
        self.assertEqual(start_date.get('type'), 'date')
        self.assertEqual(end_date.get('type'), 'date')

        # 6. Check Dual Canvas Elements
        req_canvas = soup.find('canvas', id='telemetryRequestsChart')
        tok_canvas = soup.find('canvas', id='telemetryTokensChart')
        self.assertIsNotNone(req_canvas)
        self.assertIsNotNone(tok_canvas)

        # 7. Check Educational Card Verbatim Text
        page_text = " ".join(soup.get_text().split())
        self.assertIn("Other Important Statistic not Available for Low Performance Computer:", page_text)
        self.assertIn("Time to First Token (TTFT): The duration between a user sending a prompt and receiving the very first token. This is the most critical metric for perceived speed in streaming applications.", page_text)
        self.assertIn("Inter-Token Latency (ITL): The average time elapsed between generating each subsequent token.", page_text)
        self.assertIn("Tokens Per Second (TPS): The throughput speed of the model generation (often measured per request or aggregated across the server).", page_text)
        self.assertIn("Time Per Output Token (TPOT): The total time taken to generate the response divided by the number of output tokens.", page_text)


if __name__ == '__main__':
    unittest.main()
