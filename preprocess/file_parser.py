from langchain.schema import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, SeleniumURLLoader

from utils.logger import logger


class FileParser(object):
    """
    解析不同类型的文档数据, 支持txt, pdf, docx, url 
    """
    def __init__(self, file_path, file_content=""):
        self.file_path = file_path
        self.file_content = file_content

    def string_loader(self):
        documents = Document(page_content=self.file_content, metadata={"source": self.file_path})
        return [documents]

    def txt_loader(self):
        documents = TextLoader(self.file_path, encoding='utf-8').load()
        return documents

    def pdf_loader(self):
        loader = PyPDFLoader(self.file_path)
        documents = loader.load_and_split()
        return documents

    def docx_loader(self):
        loader = Docx2txtLoader(self.file_path)
        documents = loader.load()
        return documents

    def url_loader(self):
        loader = SeleniumURLLoader(urls=[self.file_path])
        documents = loader.load()
        return documents

    def parse(self):
        logger.info(f'parsing file: {self.file_path}')
        if self.file_content:
            return self.string_loader(), 'string'
        else:
            if self.file_path.endswith(".txt"):
                return self.txt_loader(), 'txt'
            elif self.file_path.endswith(".pdf"):
                return self.pdf_loader(), 'pdf'
            elif self.file_path.endswith(".docx"):
                return self.docx_loader(), 'docx'
            elif "http" in self.file_path:
                return self.url_loader(), 'url'
            else:
                logger.error("unsupported document type!")
                return [], ''


if __name__ == '__main__':
    url = "https://learn.microsoft.com/zh-cn/dotnet/csharp/advanced-topics/reflection-and-attributes/generics-and-reflection"
    content = FileParser(url).parse()
    print(content)
