#include <bits/stdc++.h>


using namespace std;

class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> s;
        for (int num: nums) {
            s.insert(num);
        }

        int max_len = 0;
        for (int num: s) {
            if (s.count(num - 1) == 0) {
                int cur = num;
                int local_max = 1;

                while (s.count(cur + 1)) {
                    cur++;
                    local_max++;
                }

                max_len = max(max_len, local_max);
            }
        }

        return max_len;
    }
}
