#ifndef _APP_COMPONENT_HH_
#define _APP_COMPONENT_HH_

#include <memory>
#include <oatpp/core/data/mapping/ObjectMapper.hpp>
#include <oatpp/core/macro/component.hpp>
#include <oatpp/parser/json/mapping/ObjectMapper.hpp>
#include <oatpp/web/server/HttpRouter.hpp>
#include <oatpp/network/tcp/server/ConnectionProvider.hpp>
#include <oatpp/web/server/HttpConnectionHandler.hpp>
#include "ui/ui_component.hh"
#include "db/db_component.hh"

class AppComponent {
public:
    UIComponent ui_component;

    DbComponent db_component;

    /**
    * create objectMapper component
    */
    OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::data::mapping::ObjectMapper>, objectMapper)([]() {
        auto objectMapper = oatpp::parser::json::mapping::ObjectMapper::createShared();
        objectMapper->getDeserializer()->getConfig()->allowUnknownFields = false;
        return objectMapper;
    }());

    /**
    * create http router component
    */
    OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::web::server::HttpRouter>, router)([] {
        return oatpp::web::server::HttpRouter::createShared();
    }());

    /**
    * create tcp connection provider component
    */
    OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::network::ServerConnectionProvider>, serverConnectionProvider)([] {
        return oatpp::network::tcp::server::ConnectionProvider::createShared({"0.0.0.0", 8000, oatpp::network::Address::IP_4});
    }());

    /**
    * create http connection handler
    */
    OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::network::ConnectionHandler>, connectionHandler)([] {
        OATPP_COMPONENT(std::shared_ptr<oatpp::web::server::HttpRouter>, router);
        return oatpp::web::server::HttpConnectionHandler::createShared(router);
    }());
};

#endif