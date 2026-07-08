#include <bits/stdc++.h>

using namespace std;


class Solution {
private:
    std::unordered_map<string, vector<string>> tmp;
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        for (string& str: strs) {
            string ss = str;
            sort(ss.begin(), ss.end());
            tmp[ss].emplace_back(std::move(str));
        }
        vector<vector<string>> ret;
        for (auto it = tmp.begin(); it != tmp.end(); it++) {
            ret.emplace_back(std::move(it->second));
        }
        return ret;
    }
};
