# captchaAutoLabeling
滑块、点选captcha自动标注

如有侵权，请联系作者进行删除 QQ：848416881

感谢ddddocr的作者，本项目目前主要依赖其做的工作
附上地址:https://github.com/sml2h3/ddddocr
python3.10.10

poetry使用
```
pip install poetry
poetry install # 安装python依赖包

```
运行：需要先将图片放到img目录
```
cp .env.template .env
python main.py
```

如何自定义自己的model类进行二次开发：
在src/model.py文件中新增xxxModel类并继承BaseModel类
命名规范: 自定义模型名称 + Model
xxx为自定义模型名
详情请参考OnnxModel类的实现


构建和发布包：
要构建并发布您的Python包，将自动生成所需的setup.py、setup.cfg和MANIFEST.in等文件，并将包发布到PyPI
```
poetry build
poetry publish
```
