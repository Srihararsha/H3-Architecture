# H3 SSM Mathematical Visualizer & Debugger

Repository: https://github.com/Srihararsha/H3-Architecture

This project is an interactive, step-by-step tensor trace debugger and visualizer for the **H3 (Hungry Hungry Hippos)** State Space Model. It instruments the PyTorch implementation of the H3 layer, outputs the full mathematical execution trace as structured JSON chunks and a detailed markdown report, and provides a premium web dashboard to visualize the execution.

---

## 🚀 Getting Started

### 1. Environment Setup
Activate the virtual environment:
```bash
source venv/bin/activate
```

### 2. Generate the Execution Trace
To run the instrumented model and generate the mathematical trace reports:
```bash
python generate_text_h3_mathematical.py --dmodel 768 --nlayer 12 --prompt "Hi H3"
```
This script will:
- Run the H3 model text generation for the given prompt.
- Intercept embedding lookup, block inputs, projections, FFT Convolutions, and recurrent state steps.
- Output the full trace to `h3_mathematical_trace.md` (markdown format).
- Stream step data to `h3_index.js` and serialize large tensors into the `chunks/` directory as JSON files.

### 3. Start the Visualizer Dashboard
Start a local web server to host the interactive visualizer:
```bash
python -m http.server 8000
```
Then, open your browser and navigate to one of the dashboard pages:
```
http://localhost:8000/dashboard.html
```
or
```
http://localhost:8000/index.html
```

> If you regenerate `h3_index.js` or any dashboard assets, stop the server and restart it so the browser loads the newest files.

> If the page still shows old content, refresh the browser and use a hard reload / cache bypass:
> - Windows/Linux: `Ctrl+Shift+R`
> - Mac: `Cmd+Shift+R`

### 4. Finding Text Messages in the Dashboard
- Use the `Text` filter pill in the sidebar to show only text steps.
- Search for keywords like `LayerNorm`, `eps`, or `epsilon` to locate entries such as `**LayerNorm Epsilon (eps)**: `1e-05``.

---

## 🎨 Features

- **Interactive Sidebar Navigation**: Navigate through each execution step (Embedding Layer, projections, FFT convolutions, recurrent step state transitions) sequentially.
- **Vibrant Glassmorphism Design**: Curated dark-mode theme built using modern styling best practices with interactive hover effects and smooth transitions.
- **KaTeX Equation Rendering**: View exact mathematical formulas alongside computed values.
- **Tensor Shape & Statistics Inspection**: Automatically parses tensor dimensions, standard deviations, means, and roundings.
- **Dynamic Chunk Fetching**: Large tensors are streamed in real-time from the backend to ensure the dashboard remains incredibly fast and responsive.
- **Search & Filter**: Search parameters or inspect specific equations, tensors, or section markers in real-time.

---

## 📁 Repository Structure

* `generate_text_h3_mathematical.py` - The instrumented H3 model run and generation script.
* `index.html` - The main visualizer dashboard page.
* `dashboard.js` - The visualizer client-side logic.
* `h3_index.js` / `h3_index.json` - The generated step index data.
* `chunks/` - Directory containing dynamic JSON files for large tensors.
* `no_of_layers.py` - Utility script for analyzing checkpoint parameters.
