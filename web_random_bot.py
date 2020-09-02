from dlgo.agent.naive import RandomBot
from dlgo.httpfrontend.my_modify_server import get_web_app

random_agent = RandomBot()
web_app = get_web_app({'random': random_agent})
web_app.run()
