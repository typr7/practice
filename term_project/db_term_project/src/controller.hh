#ifndef _CONTROLLER_HH_
#define _CONTROLLER_HH_

#include <memory>
#include <oatpp/core/Types.hpp>
#include <oatpp/core/macro/component.hpp>
#include <oatpp/parser/json/mapping/ObjectMapper.hpp>
#include <oatpp/web/server/api/ApiController.hpp>
#include <string>
#include <iostream>
#include "service/quzl_service.hh"
#include "service/user_service.hh"
#include "dto/quzl_dto.hh"
#include "ui/resource.hh"
#include "utils.hh"

class Controller: public oatpp::web::server::api::ApiController {
public:
    Controller(const std::shared_ptr<ObjectMapper>& objectMapper,
            const std::shared_ptr<Resource>& resource);

private:
    UserService user_service_;
    QuzlService quzl_service_;

    std::shared_ptr<Resource> resource_;

public:

    static std::shared_ptr<Controller> createShared(
            OATPP_COMPONENT(std::shared_ptr<ObjectMapper>, objectMapper),
            OATPP_COMPONENT(std::shared_ptr<Resource>, resource)) {
        return std::make_shared<Controller>(objectMapper, resource);
    }

#include OATPP_CODEGEN_BEGIN(ApiController) // codegen begin

    ENDPOINT("POST", "/login", login,
             BODY_DTO(Object<UserDto>, user)) {
        auto session_id = user_service_.checkUser(user);
        auto ret = createResponse(Status::CODE_200, "登录成功");
        ret->putHeader("Set-Cookie", "session_id=" + std::to_string(session_id) + "; path=/;");
        return ret;
    }

    ENDPOINT("POST", "/user/signin", signIn,
             BODY_DTO(Object<UserDto>, user)) {
        user_service_.createUser(user);
        
        oatpp::Vector<oatpp::Object<UserDto>> vec;
        return createResponse(Status::CODE_200, "注册成功");
    }

    ENDPOINT("GET", "/user/username", userName, HEADER(oatpp::String, cookie, "Cookie")) {
        int64_t session_id = getCookieVal(cookie, "session_id");
        auto user = user_service_.getUserByCookie(session_id);
        return createDtoResponse(Status::CODE_200, user);
    }

    ENDPOINT("GET", "/user/quit", quitLogin, HEADER(oatpp::String, cookie, "Cookie")) {
        int64_t session_id = getCookieVal(cookie, "session_id");
        user_service_.deleteSessionId(session_id);
        return createResponse(Status::CODE_200);
    }

    /*
    ENDPOINT("POST", "/user/test", test, BODY_DTO(Object<QuzlDto>, quzl)) {
        using namespace std;
        cout << quzl->quzl_name->c_str() << "\n";
        cout << quzl->quz_list[0]->quz_content->c_str() << "\n";
        cout << quzl->quz_list[0]->option_list[0]->option_content->c_str() << "\n";

        return createResponse(Status::CODE_200, "ok");
    }
    */

    ENDPOINT("POST", "/quzl/create", createQuzlist,
            BODY_DTO(Object<QuzlDto>, quzl),
            HEADER(oatpp::String, cookie, "Cookie")) {
        int64_t session_id = getCookieVal(cookie, "session_id");
        auto user = user_service_.getUserByCookie(session_id);
        quzl_service_.createQuzlist(quzl, user->id);
        return createResponse(Status::CODE_200, "创建成功");
    }

    ENDPOINT("GET", "/quzl/all", getAllQuzl) {
        return createDtoResponse(Status::CODE_200, quzl_service_.getAllQuzl());
    }

    ENDPOINT("GET", "/quzl/getquzl/id/{id}",
            getCookieById, PATH(String, id)) {
        auto ret = createResponse(Status::CODE_200);
        ret->putHeader("Set-Cookie", "quzlid=" + id + "; path=/quzl/getquzl/;");
        return ret;
    }

    ENDPOINT("GET", "/quzl/getquzl/get",
            getQuzlData, HEADER(oatpp::String, cookie, "Cookie")) {
        int64_t quzl_id = getCookieVal(cookie, "quzlid");
        auto quzl = quzl_service_.getQuzlById(quzl_id);
        return createDtoResponse(Status::CODE_200, quzl);
    }

#include OATPP_CODEGEN_END(ApiController) // codegen end

};

#endif