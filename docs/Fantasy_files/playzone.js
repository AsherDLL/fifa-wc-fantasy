(function() {
    var currentScript = document.currentScript;
    var url = new URL(document.currentScript.src);
    var lang = document.currentScript.dataset.language || "";
    var splitPath = window.location.pathname.split("/");
    var game = document.currentScript.dataset.game || null;
    var pageUrl = new URL(window.location.href);
    var app = document.currentScript.dataset.app ?? pageUrl.searchParams.get("app");

    var Widget = {
        widgetData: {},
        init: function () {
            this.initWidgetData();
            this.initWidgetView();
        },
        initWidgetData: function () {
            this.checkDataWithError(currentScript, "Invalid script element");
            
            var widgetWidth = "100%";
            var widgetHeight = "272px";

            this.widgetData.element = currentScript.parentNode;
            this.widgetData.width = widgetWidth;
            this.widgetData.height = widgetHeight;
            this.widgetData.widgetElement = document.createElement("iframe");
        },
        initWidgetView: function () {
            var domain = url.origin + "/widgets/" + this.getLanguage() + "/play-zone";

            if (game) {
                domain += "/" + game;
            }

            var params = new URLSearchParams();

            if (app) {
                params.append("app", app);
            }

            if (params.toString()) {
                domain += "?" + params.toString();
            }

            this.widgetData.widgetElement.setAttribute("src", domain);
            this.widgetData.widgetElement.setAttribute("frameborder", "0");
            this.widgetData.widgetElement.setAttribute("width", String(this.widgetData.width));
            this.widgetData.widgetElement.setAttribute("height", String(this.widgetData.height));
            this.widgetData.widgetId = this.generateId();
            this.widgetData.widgetElement.setAttribute("id",this.widgetData.widgetId);
            this.widgetData.element?.appendChild(this.widgetData.widgetElement);
        },
        generateId: function () {
            return "widget_" + Math.floor(Math.random() * 1000);
        },
        getLanguage: function () {
            var supportedLanguages = ["en", "es", "fr", "de", "ar", "pt", "it", "id", "ko", "ja"];
            
            if (supportedLanguages.includes(lang)) {
                return lang;
            }
            
            var pathLanguage = splitPath.find((path) => supportedLanguages.includes(path));
            if (pathLanguage) {
                return pathLanguage;
            }

            return "en";
        },
        checkDataWithError: function (item, message) {
            if (!item) {
                this.throwError(message);
            }
        },
        throwError: function (message) {
            throw new Error(message);
        }
    };

    if (document.readyState === "complete" || document.readyState === "interactive") {
        Widget.init();
    } else {
        window.addEventListener('load', () => {
            Widget.init();
        });
    }
})();