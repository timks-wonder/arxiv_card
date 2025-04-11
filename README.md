# arXiv Cards - 论文推荐系统

## 项目简介

一个基于Flask的arXiv论文推荐系统，使用语义嵌入(Sentence Transformers)和协同过滤技术，为用户提供个性化的计算机视觉领域论文推荐服务。

## 核心功能

### 用户系统
- 用户注册/登录
- 会话管理
- 退出登录

### 推荐系统
- 基于用户兴趣的论文召回(recall)
- 基于余弦相似度的论文排序(rank)
- 用户反馈实时更新兴趣模型
  - 👍 喜欢 - 正向调整用户embedding
  - 👎 不喜欢 - 负向调整用户embedding

### 论文展示
- 论文卡片式布局
- 摘要折叠/展开功能
- 作者/分类/发布时间信息展示
- 直接链接到arXiv原文

## 技术架构

### 后端技术
- Python 3.x
- Flask + Flask-CORS
- Pandas + NumPy
- Sentence Transformers (all-MiniLM-L6-v2)
- arxiv.py (论文爬取)

### 前端技术
- Bootstrap 5
- Vanilla JavaScript
- 响应式设计

### 算法模型
- 召回阶段：基于标题embedding的余弦相似度
- 排序阶段：基于摘要embedding的余弦相似度
- 用户兴趣更新：
  - 喜欢：`user_emb * 0.9 + paper_emb * 0.1`
  - 不喜欢：`user_emb * 1.01 - paper_emb * 0.01`

## 安装指南

1. 克隆仓库：
```bash
git clone https://github.com/your-repo/arxiv_card.git
cd arxiv_card