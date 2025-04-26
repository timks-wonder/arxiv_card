import subprocess
import os
from concurrent.futures import ThreadPoolExecutor

# 定义要运行的脚本列表
scripts = [
    "create_papers_ai.py",
    # "create_papers_cv.py",
    "create_papers_cl.py",
    "create_papers_lg.py",
    "create_papers_ar.py",
    "create_papers_cc.py",
    "create_papers_ce.py",
    "create_papers_cg.py",
    "create_papers_cr.py",
    "create_papers_cy.py",
    "create_papers_db.py",
    "create_papers_dc.py",
    "create_papers_dl.py",
    "create_papers_dm.py",
    "create_papers_ds.py",
    "create_papers_et.py",
    "create_papers_fl.py",
    "create_papers_gl.py",
    "create_papers_gr.py",
    "create_papers_gt.py",
    "create_papers_hc.py",
]

def run_script(script):
    """在conda环境中运行单个脚本"""
    try:
        cmd = f"conda run -n proj2 python {script}"
        print(f"开始执行: {script}")
        subprocess.run(cmd, shell=True, check=True)
        print(f"完成执行: {script}")
    except subprocess.CalledProcessError as e:
        print(f"执行 {script} 时出错: {e}")

if __name__ == "__main__":
    # 确保当前目录正确
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 使用线程池并行执行
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(run_script, scripts)
    
    print("所有脚本执行完毕")