import logging
 
logger = logging.getLogger()
logger.setLevel(logging.INFO)   # 设置打印级别
formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')
 
# 设置屏幕打印的格式
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)
 
# 设置log保存
fh = logging.FileHandler("test.txt", encoding='utf-8')
fh.setFormatter(formatter)
logger.addHandler(fh)

logging.info('Start print log......')
