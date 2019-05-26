# -*- coding:utf-8 -*-
__author__ = 'yangxin_ryan'
from numpy import *
from votesmart import votesmart


class Apriori(object):

    def load_data_set(self):
        return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

    def create_c1(self, data_set):
        # 创建集合C1,即对data_set去重、排序、放入list中
        # 然后转换所有的元素为 frozenset
        c1 = []
        for transaction in data_set:
            for item in transaction:
                if not [item] in c1:
                    c1.append([item])
        c1.sort()
        return map(frozenset, c1)

    def scan_d(self, d, ck, min_support):
        # 计算候选数据集 CK 在数据集 D 中的支持度，并返回支持度大于最小支持度（minSupport）的数据
        ss_cnt = {}
        for tid in d:
            for can in ck:
                if can.issubset(tid):
                    if not ss_cnt.has_key(can):
                        ss_cnt[can] = 1
                    else:
                        ss_cnt[can] += 1
        num_items = float(len(d))
        ret_list = []
        support_data = {}
        for key in ss_cnt:
            support = ss_cnt[key] / num_items
            if support >= min_support:
                ret_list.insert(0, key)
            support_data[key] = support_data
        return ret_list, support_data

    def apriori_gen(self, lk, k):
        # 计算候选数据集 CK 在数据集 D 中的支持度，并返回支持度大于最小支持度（minSupport）的数据
        ret_list = []
        len_lk = len(lk)
        for i in range(len_lk):
            for j in range(i + 1, len_lk):
                l1 = list(lk[i])[: k - 2]
                l2 = list(lk[j])[: k - 2]
                l1.sort()
                l2.sort()
                if l1 == l2:
                    ret_list.append(lk[i] | lk[j])
        return ret_list

    def apriori(self, data_set, min_support=0.5):
        # 找出数据集 dataSet 中支持度 >= 最小支持度的候选项集以及它们的支持度。即我们的频繁项集。
        c1 = self.create_c1(data_set)
        d = map(set, data_set)
        l1, support_data = self.scan_d(d, c1, min_support)
        l = [l1]
        k = 2
        while (len(l[k - 2]) > 0):
            ck = self.apriori_gen(l[k - 2], k)
            lk, sup_k = self.scan_d(d, ck, min_support)
            support_data.update(sup_k)
            if len(lk) == 0:
                break
            l.append(lk)
            k += 1
        return l, support_data

    def calc_conf(self, freq_set, h, support_data, brl, min_conf=0.7):
        # 计算可信度（confidence）
        prune_h = []
        for conseq in h:
            conf = support_data[freq_set] / support_data[freq_set-conseq]
            if conf >= min_conf:
                print(freq_set - conseq, '-->', conseq, 'conf:', conf)
                brl.append((freq_set - conseq, conseq, conf))
                prune_h.append(conseq)
        return prune_h

    def rules_from_conseq(self, freq_set, h, support_data, brl, min_conf=0.7):
        # 递归计算频繁项集的规则
        m = len(h[0])
        if len(freq_set) > (m + 1):
            hmp1 = self.apriori_gen(h, m + 1)
            hmp1 = self.calc_conf(freq_set, hmp1, support_data, brl, min_conf)
            if len(hmp1) > 1:
                self.rules_from_conseq(freq_set, hmp1, support_data, brl, min_conf)

    def generate_rules(self, l, support_data, min_conf=0.7):
        # 递归计算频繁项集的规则
        big_rule_list = []
        for i in range(1, len(l)):
            for freq_set in l[i]:
                h1 = [frozenset([item]) for item in freq_set]
                if i > 1:
                    self.rules_from_conseq(freq_set, h1, support_data, big_rule_list, min_conf)
                else:
                    self.calc_conf(freq_set, h1, support_data, big_rule_list, min_conf)
        return big_rule_list

    def get_action_ids(self):
        from time import sleep
        votesmart.apikey = "xxxx"
        action_id_list = []
        bill_title_list = []
        fr = open('xxx/xxxx.test.txt')
        for line in fr.readlens():
            bill_num = int(line.split('\t')[0])
            try:
                bill_detail = votesmart.votes.get_bill(bill_num)
                for action in bill_detail.actions:
                    if action.level == 'House' and (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                        action_id = int(action.actionId)
                        print('bill: %d has actionId: %d' % (bill_num, action_id))
                        action_id_list.append(action_id)
                        bill_title_list.append(line.strip().split('\t')[1])
            except:
                print("problem getting bill %d" % bill_num)
            sleep(1)  # delay to be polite
            return action_id_list, bill_title_list

    def get_trans_list(self, action_id_list, bill_title_list):
        item_meaning = ['republican', 'democratic']
        for bill_title in bill_title_list:
            item_meaning.append('%s -- Nay' % bill_title)
            item_meaning.append('%s -- Yea' % bill_title)
        trans_dict = {}
        vote_count = 2
        for actionId in action_id_list:
            sleep(3)
            print('getting votes for actionId: %d' % actionId)
            try:
                voteList = votesmart.votes.getBillActionVotes(actionId)
                for vote in voteList:
                    if not trans_dict.has_key(vote.candidateName):
                        trans_dict[vote.candidateName] = []
                        if vote.officeParties == 'Democratic':
                            trans_dict[vote.candidateName].append(1)
                        elif vote.officeParties == 'Republican':
                            trans_dict[vote.candidateName].append(0)
                    if vote.action == 'Nay':
                        trans_dict[vote.candidateName].append(vote_count)
                    elif vote.action == 'Yea':
                        trans_dict[vote.candidateName].append(vote_count + 1)
            except:
                print("problem getting actionId: %d" % actionId)
            vote_count += 2
        return trans_dict, item_meaning

    def test_apriori(self):
        # 加载测试数据集
        data_set = self.load_data_set()
        print('dataSet: ', data_set)

        # Apriori 算法生成频繁项集以及它们的支持度
        l1, support_data1 = self.apriori(data_set, min_support=0.7)
        print('L(0.7): ', l1)
        print('supportData(0.7): ', support_data1)
        l2, support_data2 = self.apriori(data_set, min_support=0.5)
        print('L(0.5): ', l2)
        print('supportData(0.5): ', support_data2)

    def test_generate_rules(self):
        # 加载测试数据集
        data_set = self.load_data_set()
        print('dataSet: ', data_set)
        # Apriori 算法生成频繁项集以及它们的支持度
        l1, support_data1 = self.apriori(data_set, min_support=0.5)
        print('L(0.7): ', l1)
        print('supportData(0.7): ', support_data1)

        # 生成关联规则
        rules = self.generate_rules(l1, support_data1, min_conf=0.5)
        print('rules: ', rules)

    def main(self):
        # 测试 Apriori 算法
        # testApriori()

        # 生成关联规则
        # testGenerateRules()

        ##项目案例
        # # 构建美国国会投票记录的事务数据集
        # actionIdList, billTitleList = getActionIds()
        # # 测试前2个
        # transDict, itemMeaning = getTransList(actionIdList[: 2], billTitleList[: 2])
        # transDict 表示 action_id的集合，transDict[key]这个就是action_id对应的选项，例如 [1, 2, 3]
        # transDict, itemMeaning = getTransList(actionIdList, billTitleList)
        # # 得到全集的数据
        # dataSet = [transDict[key] for key in transDict.keys()]
        # L, supportData = apriori(dataSet, minSupport=0.3)
        # rules = generateRules(L, supportData, minConf=0.95)
        # print (rules)

        # # 项目案例
        # # 发现毒蘑菇的相似特性
        # # 得到全集的数据
        data_set = [line.split() for line in open("xxxxxdat").readlines()]
        l, support_data = self.apriori(data_set, min_support=0.3)
        # # 2表示毒蘑菇，1表示可食用的蘑菇
        # # 找出关于2的频繁子项出来，就知道如果是毒蘑菇，那么出现频繁的也可能是毒蘑菇
        for item in l[1]:
            if item.intersection('2'):
                print(item)

        for item in l[2]:
            if item.intersection('2'):
                print(item)


if __name__ == "__main__":
    apriori = Apriori()
    apriori.main()
