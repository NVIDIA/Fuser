class ProxyHandler {
  name = "proxy-handler";

  constructor(options) {
    this.options = options;
  }

  async transformRequestIn(request, provider) {
    // delete request.stream;
    if (request.thinking) {
        request.thinking = {
            "type": "enabled",
            "budget_tokens": 4000
        }
    }
    if (request.enable_thinking) {
        delete request.enable_thinking;
    }
    if (request.reasoning) {
        delete request.reasoning;
    }

    return {
      body: request,
    };
  }
}

module.exports = ProxyHandler
