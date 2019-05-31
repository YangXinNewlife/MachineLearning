# -*- coding:utf-8 -*-
__author__ = 'yangxin_ryan'
"""
Frequent Patten 
"""


class FPGrowth(object):

    def __init__(self, name_value, num0_ccur, parent_node):
        self.name = name_value
        self.count = num0_ccur
        self.node_link = None
        self.parent = parent_node
        self.children = {}

    def inc(self, num0_ccur):
        self.count += num0_ccur

    def disp(self, ind=1):
        print(' ' * ind, self.name, ' ', self.count)
        for chile in self.children.values():
            chile.disp(ind + 1)

    def load_simp_dat(self):
        simp_dat = [['r', 'z', 'h', 'j', 'p'],
                   ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                   ['z'],
                   ['r', 'x', 'n', 'o', 's'],
                   #    ['r', 'x', 'n', 'o', 's'],
                   ['y', 'r', 'x', 'z', 'q', 't', 'p'],
                   ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
        return simp_dat

    def create_init_set(self, data_set):
        ret_dict = {}
        for trans in data_set:
            if frozenset(trans) not in ret_dict.keys():
                ret_dict[frozenset(trans)] = 1
            else:
                ret_dict[frozenset(trans)] += 1
        return ret_dict

    def update_header(self, node_to_test, target_node):
        while node_to_test.node_link is not None:
            node_to_test = node_to_test.node_link
        node_to_test.node_link = target_node

    def update_tree(self, items, in_tree, header_table, count):
        if items[0] in in_tree.children:
            in_tree.children[items[0]].inc(count)
        else:
            in_tree.children[items[0]] = tree_node(items[0], count, in_tree)
            if header_table[items[0]][1] is None:
                header_table[items[0]][1] = in_tree.children[items[0]]
            else:
                self.update_header(header_table[items[0]][1], in_tree.children[items[0]])
        if len(items) > 1:
            self.update_tree(items[1:], in_tree.children[items[0]], header_table, count)

    def create_tree(self, data_set, min_sup=1):
        header_table = {}
        for trans in data_set:
            for item in trans:
                header_table[item] = header_table.get(item, 0) + data_set[trans]
        for k in list(header_table.keys()):
            if header_table[k] < min_sup:
                del(header_table[k])

        freq_item_set = set(header_table.keys())
        if len(freq_item_set) == 0:
            return None, None
        for k in header_table:
            header_table[k] = [header_table[k], None]
        ret_tree = tree_node('Null set', 1, None)
        for tran_set, count in data_set.items():
            local_d = {}
            for item in tran_set:
                if item in freq_item_set:
                    local_d[item] = header_table[item][0]
            if len(local_d) > 0:
                ordered_items = [v[0] for v in sorted(local_d.items(), key=lambda p: p[1], reverse=True)]
                self.update_tree(ordered_items, ret_tree, header_table, count)
        return ret_tree, header_table

    def ascend_tree(self, leaf_node, prefix_path):
        if leaf_node.parent is not None:
            prefix_path.append(leaf_node.name)
            self.ascend_tree(leaf_node.parent, prefix_path)

    def find_prefix_path(self, base_pat, tree_node):
        cond_pats = {}
        while tree_node is not None:
            prefix_path = []
            self.ascend_tree(tree_node, prefix_path)
            if len(prefix_path) > 1:
                cond_pats[frozenset(prefix_path[1:])] = tree_node.count
            tree_node = tree_node.node_link
        return cond_pats

    def mine_tree(self, in_tree, head_table, min_sup, pre_fix, freq_item_list):
        big_l = [v[0] for v in sorted(head_table.items(), key=lambda p: p[1][0])]
        for base_pat in big_l:
            new_freq_set = pre_fix.copy()
            new_freq_set.add(base_pat)
            freq_item_list.append(new_freq_set)
            cond_patt_based = self.find_prefix_path(base_pat, head_table[base_pat](1))
            my_cond_tree, my_head = self.create_tree(cond_patt_based, min_sup)
            if my_head is not None:
                my_cond_tree.disp(1)
                self.mine_tree(my_cond_tree, my_head, min_sup, new_freq_set, freq_item_list)


if __name__ == "__main__":
    fp_grouwth = FPGrowth()
    simp_dat = fp_grouwth.load_simp_dat()
    init_set = fp_grouwth.create_init_set(simp_dat)
    my_fp_tree, my_header_tab = fp_grouwth.create_tree(init_set, 3)
    my_fp_tree.disp()
    freq_item_list = []
    fp_grouwth.mine_tree(my_fp_tree, my_header_tab, 3, set([]), freq_item_list)
