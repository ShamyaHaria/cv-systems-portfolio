"""
Author: Shamya Haria
Date: February 5, 2026
Purpose: Web-based GUI for Content-Based Image Retrieval system using Flask
"""

from flask import Flask, request, jsonify, send_file
import subprocess
import os

app = Flask(__name__)

DATABASE_DIR = os.path.abspath("../../data/olympus")
BUILD_DIR = os.path.abspath("../../build")

@app.route('/')
def index():
    """Serve the main HTML interface"""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Image Retrieval System</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }
        
        .header h1 {
            font-size: 2.8em;
            font-weight: 300;
            margin-bottom: 10px;
            letter-spacing: -1px;
        }
        
        .header .icon {
            font-size: 3em;
            margin-bottom: 15px;
        }
        
        .header p {
            font-size: 1em;
            opacity: 0.9;
            font-weight: 300;
        }
        
        .card {
            background: white;
            border-radius: 16px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            margin-bottom: 30px;
        }
        
        .form-grid {
            display: grid;
            grid-template-columns: 2fr 1.5fr 100px;
            gap: 20px;
            align-items: end;
            margin-bottom: 25px;
        }

        .search-button-wrapper {
            text-align: center;
            margin-top: 10px;
        }

        .search-button-wrapper button {
            width: 300px;
        }
        
        .form-group {
            display: flex;
            flex-direction: column;
        }
        
        .form-group label {
            font-size: 13px;
            font-weight: 600;
            color: #555;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        input, select {
            padding: 14px 16px;
            border: 2px solid #e5e7eb;
            border-radius: 10px;
            font-size: 15px;
            transition: all 0.3s ease;
            background: #fafafa;
        }
        
        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
            background: white;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        input[type="number"] {
            text-align: center;
            font-weight: 600;
        }
        
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 14px 32px;
            border-radius: 10px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        #loading {
            display: none;
            text-align: center;
            padding: 80px 40px;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }

        #loading .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
        }

        #loading .loading-text {
            color: #333;
            margin-top: 15px;
        }
        
        .spinner {
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-top: 4px solid white;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .loading-text {
            color: white;
            font-size: 18px;
            font-weight: 300;
        }
        
        .results-header {
            text-align: center;
            margin-bottom: 35px;
            padding-bottom: 25px;
            border-bottom: 2px solid #f0f0f0;
        }
        
        .results-header h2 {
            font-size: 1.8em;
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 10px;
        }
        
        .results-info {
            color: #718096;
            font-size: 14px;
        }
        
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 25px;
        }
        
        .result-card {
            background: #fafafa;
            border-radius: 12px;
            overflow: hidden;
            transition: all 0.3s ease;
            border: 2px solid transparent;
        }
        
        .result-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.15);
            border-color: #667eea;
        }
        
        .result-card img {
            width: 100%;
            height: 280px;
            object-fit: cover;
            display: block;
        }
        
        .result-info {
            padding: 18px;
            text-align: center;
        }
        
        .badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .target-badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .match-badge {
            background: #10b981;
            color: white;
        }
        
        .distance {
            margin-top: 8px;
            font-size: 13px;
            color: #6b7280;
            font-family: 'Courier New', monospace;
        }
        
        #resultsContainer {
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="icon">🔍</div>
            <h1>Content Based Image Retrieval</h1>
            <p>CS 5330 - Pattern Recognition & Computer Vision | Northeastern University</p>
            <p>by Shamya Haria</p>
        </div>
        
        <div class="card">
            <div class="form-grid">
                <div class="form-group">
                    <label>Target Image Path</label>
                    <input type="text" id="targetImage" placeholder="e.g., ../../data/olympus/pic.0164.jpg" value="../../data/olympus/pic.0164.jpg">
                </div>
                
                <div class="form-group">
                    <label>Matching Method</label>
                    <select id="method">
                        <option value="baseline">Baseline (7×7 SSD)</option>
                        <option value="histogram" selected>Histogram (rg Chromaticity)</option>
                        <option value="multi_histogram">Multi-Histogram (Spatial)</option>
                        <option value="texture_color">Texture + Color</option>
                        <option value="dnn">DNN Embeddings</option>
                        <option value="adaptive">Adaptive Weighting ⭐</option>
                        <option value="saliency">Saliency-Based ⭐</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>Results</label>
                    <input type="number" id="numResults" value="5" min="1" max="10">
                </div>
            </div>
            
            <div class="search-button-wrapper">
                <button onclick="search()">🔍 Search Similar Images</button>
            </div>
        </div>
        
        <div id="loading" class="card">
            <div class="spinner"></div>
            <p class="loading-text">Searching database...</p>
        </div>
        
        <div id="resultsContainer">
            <div class="card">
                <div class="results-header">
                    <h2>Search Results</h2>
                    <p id="resultsInfo" class="results-info"></p>
                </div>
                <div id="results" class="results-grid"></div>
            </div>
        </div>
    </div>
    
    <script>
        async function search() {
            const targetImage = document.getElementById('targetImage').value;
            const method = document.getElementById('method').value;
            const numResults = parseInt(document.getElementById('numResults').value);
            
            if (!targetImage) {
                alert('Please enter a target image path');
                return;
            }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('resultsContainer').style.display = 'none';
            
            try {
                const response = await fetch('/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        target_image: targetImage,
                        method: method,
                        num_results: numResults
                    })
                });
                
                const data = await response.json();
                document.getElementById('loading').style.display = 'none';
                
                if (data.success) {
                    displayResults(targetImage, data.results, method);
                    document.getElementById('resultsContainer').style.display = 'block';
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                document.getElementById('loading').style.display = 'none';
                alert('Error: ' + error);
            }
        }
        
        function displayResults(targetImage, results, method) {
            const container = document.getElementById('results');
            container.innerHTML = '';
            
            const methodNames = {
                'baseline': 'Baseline Matching',
                'histogram': 'Histogram Matching',
                'multi_histogram': 'Multi-Histogram Matching',
                'texture_color': 'Texture + Color',
                'dnn': 'DNN Embeddings (ResNet18)',
                'adaptive': 'Adaptive Feature Weighting',
                'saliency': 'Saliency-Based Matching'
            };
            
            document.getElementById('resultsInfo').innerHTML = 
                'Method: <strong>' + methodNames[method] + '</strong> | Found <strong>' + results.length + '</strong> similar images';
            
            const targetCard = document.createElement('div');
            targetCard.className = 'result-card';
            targetCard.innerHTML = 
                '<img src="/image?path=' + encodeURIComponent(targetImage) + '" alt="Target">' +
                '<div class="result-info"><span class="badge target-badge">Target Image</span></div>';
            container.appendChild(targetCard);
            
            results.forEach((result, i) => {
                const card = document.createElement('div');
                card.className = 'result-card';
                card.innerHTML = 
                    '<img src="/image?path=' + encodeURIComponent(result.path) + '" alt="Match ' + (i+1) + '">' +
                    '<div class="result-info">' +
                    '<span class="badge match-badge">Match #' + (i + 1) + '</span>' +
                    '<div class="distance">Distance: ' + result.distance.toFixed(6) + '</div>' +
                    '</div>';
                container.appendChild(card);
            });
        }
    </script>
</body>
</html>
    '''

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests by invoking C++ executables and parsing results"""
    data = request.json
    target = data.get('target_image')
    method = data.get('method', 'histogram')
    num = data.get('num_results', 5)
    
    # Map method names to executable files
    methods = {
        "baseline": "baseline_matching",
        "histogram": "histogram_matching",
        "multi_histogram": "multi_histogram",
        "texture_color": "texture_color",
        "dnn": "dnn_matching",
        "adaptive": "adaptive_matching",
        "saliency": "saliency_matching"
    }
    
    exe = os.path.abspath(os.path.join(BUILD_DIR, methods[method]))
    target_abs = os.path.abspath(target)
    
    # Build command based on method type
    if method == "dnn":
        embeddings = os.path.abspath("../../data/embeddings.csv")
        cmd = [exe, target_abs, embeddings, str(num)]
    else:
        cmd = [exe, target_abs, DATABASE_DIR, str(num)]
    
    try:
        # Execute C++ program
        result = subprocess.run(cmd, capture_output=True, text=True)
        results = []
        
        # Parse output to extract image paths and distances
        for line in result.stdout.split('\n'):
            if '. ' in line and 'pic.' in line and 'distance:' in line:
                parts = line.split()
                img_path = None
                distance = None
                
                # Extract image path
                for i, part in enumerate(parts):
                    if 'pic.' in part and '.jpg' in part:
                        img_path = part.strip('(),')
                        if not os.path.isabs(img_path):
                            possible_paths = [
                                os.path.join(DATABASE_DIR, os.path.basename(img_path)),
                                os.path.join(DATABASE_DIR, img_path),
                                img_path
                            ]
                            for p in possible_paths:
                                if os.path.exists(p):
                                    img_path = os.path.abspath(p)
                                    break
                    
                    # Extract distance value
                    if part == '(distance:' and i+1 < len(parts):
                        distance = parts[i+1].strip(')')
                
                if img_path and os.path.exists(img_path):
                    try:
                        dist_val = float(distance) if distance else 0.0
                        results.append({'path': img_path, 'distance': dist_val})
                    except:
                        results.append({'path': img_path, 'distance': 0.0})
        
        return jsonify({'success': True, 'results': results[:num]})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/image')
def get_image():
    """Serve image files to the frontend"""
    path = request.args.get('path')
    if path and os.path.exists(path):
        return send_file(path, mimetype='image/jpeg')
    return '', 404

if __name__ == '__main__':
    print("\n" + "="*70)
    print("  🔍 Content-Based Image Retrieval System - Web Interface")
    print("="*70)
    print("\n  ➜ Local:   http://127.0.0.1:5000")
    print("\n" + "="*70 + "\n")
    app.run(debug=True, port=5000, host='127.0.0.1')