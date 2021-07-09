# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from snownlp import SnowNLP
import re
import numpy as np
import matplotlib.pyplot as plt
import jieba
from wordcloud import WordCloud
from sklearn.metrics import classification_report

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def load_data_sample(input_file_name, label):
    input_file = open(input_file_name, 'r', encoding='utf-8')
    id_list = []
    str_list = []
    label_list_out = []
    doc_list = []
    lines_1 = input_file.read().split('\n')
    count = 0
    last = ''
    for i in lines_1:
        if '<review id=' in i:
            ind = i.index("=")
            new_id = i[ind + 2:-2]
            id_in = int(new_id)
            id_list.append(id_in)
            label_list_out.append(label)
        else:
            if '</review>' in i:
                new_str = ''
                for j in last.split("\n"):
                    new_str += j
                new_str = re.sub("[a-zA-Z0-9]", "", new_str)
                if new_str == []:
                    label_list_out.pop()
                    id_list.pop()
                else:
                    doc_list.append(new_str)
                last = ''
            else:
                last += i
    print(doc_list[3])
    return id_list, doc_list, label_list_out

def load_data_test(input_file_name_in):
    input_file = open(input_file_name_in, 'r', encoding='utf-8')
    id_list_out = []
    doc_list = []
    label_list_out = []
    str_list = []
    lines_1 = input_file.read().split('\n')
    count = 0
    last = ''
    print('loading:')
    for i in lines_1:
        count += 1
        print('\r', count/len(lines_1)*100, '%', end='')
        if '<review id=' in i:
            isp = i.split(' ')
            for sp in isp:
                if 'id' in sp:
                    id_ind = sp.index("=")
                    new_id = sp[id_ind + 2:-1]
                    id_in = int(new_id)
                    id_list_out.append(id_in)
                elif 'label' in sp:
                    lb_ind = sp.index("=")
                    new_lb = sp[lb_ind + 2:-2]
                    label_in = int(new_lb)
                    label_list_out.append(label_in)
        else:
            if '</review>' in i:
                new_str = ''
                for j in last.split("\n"):
                    new_str += j
                new_str = re.sub("[a-zA-Z0-9]", "", new_str)
                if new_str == []:
                    label_list_out.pop()
                    id_list_out.pop()
                else:
                    doc_list.append(new_str)
                last = ''
            else:
                last += i
    id_list_output = np.array(id_list_out[0:len(id_list_out)])
    label_list_output = np.array(label_list_out[0:len(label_list_out)])
    print('\n')
    print('loading successfully!\n')
    return id_list_output, doc_list, label_list_output

def word_filter(seg_list_in):
    stopword_list = get_stopwords_list()
    filter_list_out = []
    cn_reg = '^[\u4e00-\u9fa5]+$'
    for seg in seg_list_in:
        word = seg
        if word not in stopword_list and re.search(cn_reg, word):
            filter_list_out.append(word)
    return filter_list_out

def get_stopwords_list():
    stop_word_path = './cn_stopwords.txt'
    stop_words_list = []
    stopwords_file = open(stop_word_path, 'r', encoding='utf-8')
    for sw in stopwords_file.readlines():
        stop_words_list.append(sw.replace('\n', ''))
    return stop_words_list

def load_doc_sample(input_file_name, label):
    input_file = open(input_file_name, 'r', encoding='utf-8')
    id_list = []
    str_list = []
    label_list_out = []
    doc_list = []
    lines_1 = input_file.read().split('\n')
    count = 0
    last = ''
    for i in lines_1:
        if '<review id=' in i:
            ind = i.index("=")
            new_id = i[ind + 2:-2]
            id_in = int(new_id)
            id_list.append(id_in)
            label_list_out.append(label)
        else:
            if '</review>' in i:
                new_str = ''
                for j in last.split("\n"):
                    new_str += j
                new_str = re.sub("[a-zA-Z0-9]", "", new_str)
                str_list.append(new_str)
                seg_list = jieba.cut(new_str)
                filter_list = word_filter(seg_list)
                if filter_list == []:
                    label_list_out.pop()
                    id_list.pop()
                else:
                    doc_list.append(filter_list)
                last = ''
            else:
                last += i
    print(doc_list[3])
    return id_list, doc_list, label_list_out


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    id_list_test, doc_list_test, label_list_test = load_data_test('./test.label.cn.txt')
    doc_list = doc_list_test
    label_list = label_list_test
    print(len(doc_list))
    print(len(label_list))
    print(len(id_list_test))
    label_list_true = np.array(label_list[0:len(label_list)])
    label_pre = np.zeros(len(doc_list))
    for i in range(0, len(doc_list)):
        s = SnowNLP(doc_list[i])
        if float(s.sentiments) < 0.5:
            label_pre[i] = 0
        else:
            label_pre[i] = 1
    print(label_pre)
    y_true = label_list_true
    y_pred = label_pre
    print(classification_report(y_true, y_pred))
if __name__ == "__main__":
  # 重新训练模型
  sentiment.train('./neg.txt', './pos.txt')
  # 保存好新训练的模型
  sentiment.save('sentiment.marshal')



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
