# Text Embedding and Clustering

**[English](#english)** · **[简体中文](#简体中文)**

<a name="english"></a>
## English

This repo is 100% vibe-coded. I genuinely don’t understand what I wrote. If you find bugs, redundant code, or cursed styling—please don’t yell at me. Open a pull request , or just fork it and do your own thing. Thanks. 

## What this does
This toolkit will **embed** CSV text into numerical vectors, **cluster** related data points, and **visualize** the results through interactive reports and charts.

Choose your workflow: 
- Use the **Web App** for an intuitive, no-code experience
- or run the **CLI scripts** for manual control and automation.

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/cindydidi/text-embedding-and-clustering
cd text-embedding-and-clustering
pip install -r requirements.txt
```
## Web app
### Setup

Run the pipeline in a **web interface**

1. **Python 3** on your computer.
2. After cloning and installing (see [Installation](#installation) above), create a **`.env`** file in this folder (same folder as `web_app.py`) with your Google API key:
   ```
   GOOGLE_API_KEY=your_key_here
   ```
   Keep `.env` private and don’t share it.

### Run the app

```bash
streamlit run web_app.py
```

Open the URL that appears (usually **http://localhost:8501**). Upload a CSV, pick the text column if asked, then use the three buttons to run Embedding, Clustering, and Data/Visualization. Results and downloads appear at the bottom. Config (sentiment words, chart colors, etc.) can be edited in the sidebar.

## Running the scripts from the command line

1. **Step 1** — `01_generate_embeddings.py` — reads your CSV and creates embeddings.
2. **Step 2** — `02_cluster_and_extract.py` — clusters the embeddings and adds topic labels.
3. **Step 3** — `03_analyze_and_visualize.py` — compares clusters and generates reports and charts.

Run them in that order. Put your CSV in the same folder (or point to it with `--input`). For all options, run any script with `--help`, for example:

```bash
python 01_generate_embeddings.py --help
```

## If something goes wrong

- **“GOOGLE_API_KEY not found”** — Add a `.env` file in this folder with `GOOGLE_API_KEY=your_key`.
- **Missing packages** — Run `pip install -r requirements.txt`.
- **File not found (steps 2 or 3)** — Run the steps in order (01 → 02 → 03) and keep the generated files in the same folder (or use the web app, which uses a `streamlit_workspace` folder).
- **Multiple text columns** — In the web app, choose the column from the dropdown. From the command line, use `--text-column "YourColumnName"` with step 1.

<a name="简体中文"></a>
## 简体中文

这个 repo 是 vibe code 来的，本文科生是一行代码也看不懂，如有错误，redundancy 或者 style 不好的你就忍了吧，别骂我。谢谢。欢迎 pull request 或者直接 fork 自己改。

### 它能做什么
这个工具可以把 **CSV 里的文字** 转成数值向量（嵌入），再做 **聚类**，最后 **生成图表和报告**。

两种用法你可以选：
- **网页版**：不用写代码，上传文件、点按钮就能跑完全程，适合刚开始用的人。
- **命令行**：在终端里按顺序跑脚本，适合想自己调一调或做自动化的人。

### 安装
把项目克隆到本地，然后装好 package：

```bash
git clone https://github.com/cindydidi/text-embedding-and-clustering
cd text-embedding-and-clustering
pip install -r requirements.txt
```

### 用网页版

1. 电脑上装好 **Python 3**。
2. 安装好后，在这个文件夹里（和 `web_app.py` 同一层）新建一个 **`.env`** 文件，写上你的 Google API key：
   ```
   GOOGLE_API_KEY=你的密钥
   ```
   不要泄漏 `.env`，记得密钥要自己保管好。

#### 启动应用

```bash
streamlit run web_app.py
```

终端里会给出一个地址（一般是 **http://localhost:8501**）。用浏览器打开，上传你的 CSV，需要的话选一下「文本列」，然后按顺序点：嵌入 → 聚类 → 可视化。结果和下载都在页面下面，侧边栏里可以改配置（比如情感词、图表样式等）。

### 用命令行
也可以在终端里分三步跑：

1. **第一步** — `01_generate_embeddings.py` — 读 CSV，生成嵌入向量。
2. **第二步** — `02_cluster_and_extract.py` — 做聚类，给每一类打标签。
3. **第三步** — `03_analyze_and_visualize.py` — 对比不同组，生成报告和图表。

按 01 → 02 → 03 的顺序跑就行。把 CSV 放在同一文件夹里（或用 `--input` 指定路径）。想看每个脚本有什么参数，可以加 `--help`，例如：

```bash
python 01_generate_embeddings.py --help
```

### 遇到问题怎么办
- **出现「GOOGLE_API_KEY not found」** — 在这个文件夹里加一个 `.env` 文件，写上 `GOOGLE_API_KEY=你的 key`。
- **少装了 package** — 再执行一次 `pip install -r requirements.txt`。
- **第二步或第三步说找不到文件** — 先按顺序跑完 01、02，生成的文件留着；或者直接用网页版（会用 `streamlit_workspace` 这个文件夹）。
- **CSV 里有多列文字** — 网页里在下拉菜单选一下要用哪一列；命令行在第一步加上 `--text-column "列名"`。