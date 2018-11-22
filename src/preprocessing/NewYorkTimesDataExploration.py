from contextlib import closing
from xml.dom import minidom

import os
import tarfile


def read_xml(xmlfilename):
    xml_file = minidom.parse(xmlfilename)
    head_data = xml_file.getElementsByTagName('meta')
    for meta in head_data:
        if meta.attributes['name'].value == "online_sections":
            return meta.attributes['content'].value.split("; ")
    return "Not tagged - Amal"


class ArticleTopicCounter(object):
    def __init__(self):
        self.topics = {}
        self.articles = 0

    def fill_topics_articles(self, file_name):
        relevant_topics = read_xml(file_name)
        self.articles += 1
        for topic in relevant_topics:
            if topic in self.topics:
                self.topics[topic] += 1
            else:
                self.topics[topic] = 1

    def save_to_file(self):
        csv_file = 'article_topic_counts.csv'
        try:
            with open(csv_file, 'w') as csvfile:
                csvfile.write("%s,%s\n" % ("NYT Articles Count", self.articles))
                for data in counter.topics:
                    csvfile.write("%s,%s\n" % (data, counter.topics[data]))
        except IOError:
            print("I/O error")


if __name__ == '__main__':
    # iterate through nyt folder contents
    # nyt
    # --> year [1987-2007]
    #    --> Month (tgz)
    #        --> Days (folders)
    #           --> Articles (xml)
    #              --> <head>
    #                 --> <meta>
    #                   --> name="online_sections"
    #                      --> content=X
    data_folder = "/tmp/hyperpartisan_project/project/data/nyt"

    counter = ArticleTopicCounter()

    for root, dirs, files in os.walk(data_folder):
        path = root.split(os.sep)
        print((len(path) - 1) * '---', os.path.basename(root))
        for file in files:
            filename = os.path.join(root, file)
            print(len(path) * '---!', file)
            if file.endswith('.xml'):
                counter.fill_topics_articles(filename)
            elif file.endswith('.tgz'):
                with tarfile.open(filename) as archive:
                    for member in archive:
                        if member.isreg() and member.name.endswith('.xml'):  # regular xml file
                            with closing(archive.extractfile(member)) as xmlfile:
                                counter.fill_topics_articles(xmlfile)

    counter.save_to_file()
