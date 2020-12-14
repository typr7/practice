//
// Created by ca1se on 2020/12/13.
//

#include "dealmaintain.h"

Pay::Pay(Machine *pM): _pM(pM) {}//构造函数里有参数*pM，列表后面（括号）传进一个pM




void Pay::PaymentOffline(Gname g, size_t n, const std::list<std::pair<coin, size_t>>& l) const {         //const在于常量，有了常量不需要改正数据。而修改已经调用函数的参数意义不大
    _pM->TakeOutGoods(g, n);//算是Gname完成
    for(auto& it: l){
        _pM->AddCoin(it.first, it.second);//硬币种类，硬币个数
    }
    const char* timeNow = Time::GetTimeNow();       //=后面就是返回的buf这个值
    PLOG << "[" << timeNow << "]: " << "Deal finished: " << g  << " Number: " << n << " Pay way: Offline\n";   //日志，类似于cout输出但是不好直接  函数（参数） 的形式
}



void Pay::PaymentOnline(Gname g, size_t n, std::string payway) const {
    _pM->TakeOutGoods(g, n);
    const char* timeNow = Time::GetTimeNow();
    PLOG << "[" << timeNow << "]: " << "Deal finished: " << g << " Number: " << n << "Pay way: " << payway << '\n';
}


ContainUpdate::ContainUpdate(Machine *pM): _pM(pM) {}



void ContainUpdate::ReplenishGoods(Gname g, size_t n) const {
    _pM->AddGoods(g, n);//_pM指向 Machine实例，这个函数包含在实例里面
    const char* timeNow = Time::GetTimeNow();
    MLOG << "[" << timeNow << "]" << "ReplenishGoods: " << g << "Number: " << n << '\n';
}




void ContainUpdate::ReplenishCoin(coin g, size_t n)const {
    _pM->AddCoin(g, n);///_pM指向 Machine实例，这个函数包含在实例里面
    const char* timeNow = Time::GetTimeNow();
    MLOG << timeNow << "Replenishcoins: " << g << "Number: " << n << '\n';
}



const char * Time::GetTimeNow() {
    time_t t = time(nullptr);                                               //调用时间
    char buf[80] = {0};    //给时间一个空间放时间
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&t));         //调用时间
    return buf;          //不返回会没有输出
}






