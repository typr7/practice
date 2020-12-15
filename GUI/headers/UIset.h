//
// Created by ca1se on 2020/12/14.
//

#ifndef UNTITLED2_UI_WIDGET_H
#define UNTITLED2_UI_WIDGET_H

#include <QtWidgets/QPushButton>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLabel>
#include <QApplication>

//  Main Widget description class
//  Realize the part of selecting goods
class surf_widget {
public:
    QWidget* layoutWidget;
    QGridLayout* mainGridLyt;
    QPushButton* pBtnSelect0;
    QPushButton* pBtnSelect1;
    QPushButton* pBtnSelect2;
    QPushButton* pBtnSelect3;
    QPushButton* pBtnSelect4;
    QPushButton* pBtnSelect5;
    QPushButton* pBtnSelect6;
    QPushButton* pBtnSelect7;
    QPushButton* pBtnSelect8;
    QPushButton* pBtnAdmin;
    QLabel* lGoods0;
    QLabel* lGoods1;
    QLabel* lGoods2;
    QLabel* lGoods3;
    QLabel* lGoods4;
    QLabel* lGoods5;
    QLabel* lGoods6;
    QLabel* lGoods7;
    QLabel* lGoods8;

    void initUI(QWidget* widget) {
        widget->setMinimumSize(720, 960);
        widget->setMaximumSize(720, 960);
        widget->setWindowTitle(QApplication::trUtf8("自动售货机"));
        widget->setStyleSheet("background-color: rgba(255, 227, 132, 1)");


        layoutWidget = new QWidget(widget);
        layoutWidget->setGeometry(QRect(90, 180, 540, 600));

        mainGridLyt = new QGridLayout(layoutWidget);
        mainGridLyt->setContentsMargins(20, 20, 20, 20);
        mainGridLyt->setHorizontalSpacing(10);
        mainGridLyt->setVerticalSpacing(10);

        lGoods0 = new QLabel(layoutWidget);
        setUpLabel(lGoods0, "0");
        mainGridLyt->addWidget(lGoods0, 1, 0, 1, 1);
        lGoods1 = new QLabel(layoutWidget);
        setUpLabel(lGoods1, "1");
        mainGridLyt->addWidget(lGoods1, 1, 1, 1, 1);
        lGoods2 = new QLabel(layoutWidget);
        setUpLabel(lGoods2, "2");
        mainGridLyt->addWidget(lGoods2, 1, 2, 1, 1);
        lGoods3 = new QLabel(layoutWidget);
        setUpLabel(lGoods3, "3");
        mainGridLyt->addWidget(lGoods3, 3, 0, 1, 1);
        lGoods4 = new QLabel(layoutWidget);
        setUpLabel(lGoods4, "4");
        mainGridLyt->addWidget(lGoods4, 3, 1, 1, 1);
        lGoods5 = new QLabel(layoutWidget);
        setUpLabel(lGoods5, "5");
        mainGridLyt->addWidget(lGoods5, 3, 2, 1, 1);
        lGoods6 = new QLabel(layoutWidget);
        setUpLabel(lGoods6, "6");
        mainGridLyt->addWidget(lGoods6, 5, 0, 1, 1);
        lGoods7 = new QLabel(layoutWidget);
        setUpLabel(lGoods7, "7");
        mainGridLyt->addWidget(lGoods7, 5, 1, 1, 1);
        lGoods8 = new QLabel(layoutWidget);
        setUpLabel(lGoods8, "8");
        mainGridLyt->addWidget(lGoods8, 5, 2, 1, 1);

        pBtnAdmin = new QPushButton(widget);
        pBtnAdmin->setGeometry(QRect(30, 900, 85, 35));
        pBtnAdmin->setText(QApplication::trUtf8("补货"));

        pBtnAdmin = new QPushButton(widget);
        pBtnAdmin->setGeometry(QRect(30, 840, 85, 35));
        pBtnAdmin->setText(QApplication::trUtf8("补硬币"));

        pBtnSelect0 = new QPushButton(widget);
        setUpButton(pBtnSelect0, "background-color: rgba(40, 20, 10, 1)", "0");
        mainGridLyt->addWidget(pBtnSelect0, 0, 0, 1, 1);
        pBtnSelect1 = new QPushButton(widget);
        setUpButton(pBtnSelect1, "background-color: rgba(80, 20, 10, 1)", "1");
        mainGridLyt->addWidget(pBtnSelect1, 0, 1, 1, 1);
        pBtnSelect2 = new QPushButton(widget);
        setUpButton(pBtnSelect2, "background-color: rgba(40, 90, 10, 1)", "2");
        mainGridLyt->addWidget(pBtnSelect2, 0, 2, 1, 1);
        pBtnSelect3 = new QPushButton(widget);
        setUpButton(pBtnSelect3, "background-color: rgba(20, 50, 10, 1)", "3");
        mainGridLyt->addWidget(pBtnSelect3, 2, 0, 1, 1);
        pBtnSelect4 = new QPushButton(widget);
        setUpButton(pBtnSelect4, "background-color: rgba(70, 80, 10, 1)", "4");
        mainGridLyt->addWidget(pBtnSelect4, 2, 1, 1, 1);
        pBtnSelect5 = new QPushButton(widget);
        setUpButton(pBtnSelect5, "background-color: rgba(40, 50, 30, 1)", "5");
        mainGridLyt->addWidget(pBtnSelect5, 2, 2, 1, 1);
        pBtnSelect6 = new QPushButton(widget);
        setUpButton(pBtnSelect6, "background-color: rgba(70, 90, 10, 1)", "6");
        mainGridLyt->addWidget(pBtnSelect6, 4, 0, 1, 1);
        pBtnSelect7 = new QPushButton(widget);
        setUpButton(pBtnSelect7, "background-color: rgba(40, 60, 10, 1)", "7");
        mainGridLyt->addWidget(pBtnSelect7, 4, 1, 1, 1);
        pBtnSelect8 = new QPushButton(widget);
        setUpButton(pBtnSelect8, "background-color: rgba(40, 70, 70, 1)", "8");
        mainGridLyt->addWidget(pBtnSelect8, 4, 2, 1, 1);

    }

private:

    static void setUpButton(QPushButton* pBtn, const char* styleSheet, const char* pos) {
        pBtn->setObjectName(QApplication::trUtf8(pos));
        pBtn->setMaximumSize(140, 140);
        pBtn->setMinimumSize(140, 140);
        pBtn->setStyleSheet(styleSheet);
    }

    static void setUpLabel(QLabel* lb, const char* pos) {
        lb->setObjectName(QApplication::trUtf8(pos));
        lb->setMinimumSize(140, 30);
        lb->setMaximumSize(140, 30);
        lb->setWordWrap(true);
    }
};
// Sub Widget class description class
// Realize the part of purchase goods and pay
class sub_widget {
public:

};

namespace UI {
    class MainWidget: public surf_widget {};
    class SubWidget: public sub_widget {};
}


#endif //UNTITLED2_UI_WIDGET_H
