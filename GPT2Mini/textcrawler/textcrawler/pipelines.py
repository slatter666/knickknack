# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import json


class ProsecrawlerPipeline:
    def __init__(self):
        self.file_path = 'prose.json'
        self.proses = list()

    def close_spider(self, spider):
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.proses, f, indent=2, ensure_ascii=False)

    def process_item(self, item, spider):
        try:
            self.proses.append(item)
        except:
            pass


class Prose2crawlerPipeline:
    def __init__(self):
        self.file_path = 'prose2.json'
        self.proses = list()

    def close_spider(self, spider):
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.proses, f, indent=2, ensure_ascii=False)

    def process_item(self, item, spider):
        try:
            self.proses.append(item)
        except:
            pass
