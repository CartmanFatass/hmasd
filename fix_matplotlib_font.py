import matplotlib.pyplot as plt
import matplotlib
import os
import platform

def configure_chinese_font():
    """配置matplotlib支持中文的字体"""
    system = platform.system()
    
    if system == 'Windows':
        # Windows系统默认中文字体
        font_family = ['Microsoft YaHei', 'SimHei']
    elif system == 'Darwin':
        # macOS系统默认中文字体
        font_family = ['Heiti SC', 'Hiragino Sans GB']
    else:
        # Linux系统常用中文字体
        font_family = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC']
    
    # 设置默认字体
    for font in font_family:
        try:
            matplotlib.rcParams['font.sans-serif'] = [font] + matplotlib.rcParams['font.sans-serif']
            # 测试字体是否可用
            test_str = '测试'
            fig = plt.figure(figsize=(1, 1))
            plt.text(0.5, 0.5, test_str, fontsize=12, ha='center', va='center')
            plt.close(fig)
            print(f"成功配置中文字体: {font}")
            
            # 用于解决负号显示问题
            matplotlib.rcParams['axes.unicode_minus'] = False
            return True
        except:
            continue
    
    # 如果都不可用，尝试从matplotlib的fonts目录复制中文字体
    try:
        # 查找simhei.ttf并添加到matplotlib字体管理器
        from matplotlib.font_manager import FontManager, fontManager, FontProperties
        import shutil

        # 尝试在当前目录创建一个字体
        target_font = os.path.abspath(os.path.join(os.path.dirname(__file__), 'simhei.ttf'))
        
        if not os.path.exists(target_font):
            # 尝试从系统目录复制
            potential_paths = []
            if system == "Windows":
                potential_paths = [
                    r"C:\Windows\Fonts\simhei.ttf",
                    r"C:\Windows\Fonts\msyh.ttc",
                ]
            
            font_copied = False
            for font_path in potential_paths:
                if os.path.exists(font_path):
                    shutil.copy(font_path, target_font)
                    font_copied = True
                    break
            
            if not font_copied:
                print("无法在系统中找到合适的中文字体。")
                return False
        
        # 更新字体缓存
        matplotlib.font_manager._rebuild()
        
        # 设置默认字体为新添加的字体
        plt.rcParams['font.sans-serif'] = ['SimHei'] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        print("已成功添加SimHei字体到matplotlib。")
        return True
    except Exception as e:
        print(f"配置中文字体失败: {str(e)}")
        return False

if __name__ == "__main__":
    success = configure_chinese_font()
    if success:
        # 测试中文显示
        plt.figure(figsize=(8, 6))
        plt.title('中文显示测试')
        plt.xlabel('横坐标')
        plt.ylabel('纵坐标')
        plt.text(0.5, 0.5, '测试文本', fontsize=20, ha='center', va='center')
        plt.savefig('font_test.png')
        plt.close()
        print("测试图片已保存为font_test.png")
    else:
        print("中文字体配置失败。可能需要手动安装中文字体。")
