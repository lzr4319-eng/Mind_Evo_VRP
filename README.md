# Mind Evolution (Mind Evo) 🧠🚚

**Mind Evolution** 是一个基于 **大语言模型 (LLM)** 与 **进化算法 (Evolutionary Strategy)** 深度融合的智能求解系统。

本项目旨在通过模拟生物进化的“生成-批判-改进”过程，利用 LLM 强大的推理能力（Inference-time Compute）来解决复杂的 **带时间窗车辆路径规划问题 (VRPTW)**。

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-green)
![LLM](https://img.shields.io/badge/LLM-Qwen2.5--14B-purple)

## ✨ 核心特性

- **LLM 驱动的进化算子**：抛弃传统的随机变异，利用 LLM 进行语义级别的“智能变异”和“重组”。
- **批判家-作者 (Critic-Author) 循环**：引入双角色机制，LLM 先扮演“批判家”分析方案缺陷，再扮演“作者”生成改进方案。
- **岛屿模型 (Island Model)**：支持多子种群并行进化，模拟地理隔离与基因交流，防止陷入局部最优。
- **可视化交互界面**：提供直观的 Web 界面，实时展示进化过程、思维链日志 (Chain of Thought) 和最优路径视图。
- **混合架构**：结合传统运筹学约束计算（硬代码）与大模型启发式搜索（软推理），兼顾准确性与创造性。

## 🛠️ 技术栈

- **后端**: Python 3, Flask
- **LLM 推理**: 
  - 默认: [SiliconFlow (硅基流动)](https://siliconflow.cn) API (Qwen/Qwen2.5-14B-Instruct)
  - 可选: 本地 Ollama
- **前端**: HTML5, CSS3, JavaScript (原生)
- **算法**: 遗传算法 (GA) + Prompt Engineering

## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/your-username/mind-evo-vrp.git
cd mind-evo-vrp
```

### 2. 安装依赖
建议使用 Python 3.8 或更高版本。
```bash
pip install -r requirements.txt
```

### 3. 配置 API Key
本项目默认使用 SiliconFlow 的在线 API。你需要配置环境变量，或者直接修改代码中的 Key（不推荐提交到仓库）。

Windows (PowerShell):
```powershell
$env:SILICONFLOW_API_KEY = "你的API_KEY"
```

Linux/Mac:
```bash
export SILICONFLOW_API_KEY="你的API_KEY"
```

### 4. 运行系统
```bash
python app.py
```

启动后，访问浏览器 [http://127.0.0.1:5000](http://127.0.0.1:5000) 即可使用。

## 🧩 项目结构

- `app.py`: Flask Web 服务器入口。
- `api.py`: 核心逻辑，包含 LLM 调用、进化算子和岛屿模型实现。
- `genetic.py`: 传统遗传算法辅助函数。
- `templates/`: 前端页面模板。
- `static/`: 静态资源文件。

## 📄 许可证

MIT License
