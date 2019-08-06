import pyltp_model as ltp


def main():
    news = """在今天商务部举行的例行发布会上，有媒体表示：IMF近日发布的研究报告称，美国加征关税造成的成本几乎全部由美国企业承担了，但美国总统特朗普称，中国在为美国加征的关税买单。对此，新闻发言人高峰表示：美方的贸易霸凌主义最终损害的是美国自身，买单的是美国的消费者和企业。

    　　高峰表示，中美贸易不平衡，主要是由于美国出口管制等非经济因素以及储蓄率低等原因造成的，加征关税根本解决不了贸易不平衡问题。

    　　关于美国单方面加征关税的影响，高峰强调，美方的贸易霸凌做法最终损害的是美国自身，买单的是美国的消费者和企业，纽约联储经济学家最近的预测表明，美方加征关税措施，将使每个美国家庭每年平均损失831美元。

    　　高峰表示，美国一些智库的研究也显示，如果美方的措施持续下去，会导致美国的GDP增速下滑、就业和投资减少、国内物价上升，美国商品在海外的竞争力下降，已经有越来越多的美国企业、消费者感受到加征关税的影响。

    　　与此同时，高峰再次强调了中方关于中美经贸磋商的立场：中方绝不会接受任何有损国家主权和尊严的协议，在重大原则问题上，中方绝对不会让步，如果要达成协议，美方需要拿出诚意，妥善解决中方提出的核心关切，在平等相待、相互尊重的基础上继续磋商。
    """

    saying_words_path = './model_data/saying_clean.txt'
    saying_list = ltp.load_saying_words(saying_words_path)

    # text_path = 'test_chinese_news.txt'
    # sentences = load_text(text_path)
    # sents_list = sentence_splitter(sentences)

    sents_list = ltp.sentence_splitter(news)
    total_names = ltp.get_total_names(sents_list)
    # print('全部人名', total_names.items())
    print('开始提取...')

    for sent in sents_list:
        print(sent)
        opinions = ltp.extract_single_sentence(saying_list, sent, total_names)
        if opinions:
            print('提取成功')
            for (name, op) in opinions:
                print("人物：{}\n言论：{}".format(name, op))
        else:
            print('---抱歉，没有找到言论---')
        print('*' * 80)
    print('提取结束!')


if __name__ == '__main__':
    main()
