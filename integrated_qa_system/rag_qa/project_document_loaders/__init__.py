import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 导入配置
from base.config import single_config as config

# 添加配置中的路径到 sys.path
# sys.path.append(config.project_DOCUMENT_LOADERS_DIR)

# 条件导入各个文档加载器，允许在缺少依赖时继续运行
try:
    from .project_docloader import *
except ImportError:
    pass

try:
    from .project_pptloader import *
except ImportError:
    pass

try:
    from .project_imgloader import *
except ImportError:
    pass

try:
    from .project_pdfloader import *
except ImportError:
    pass
