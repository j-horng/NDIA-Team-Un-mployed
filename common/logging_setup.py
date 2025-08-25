import logging, sys, json, time
class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({"t":int(time.time()*1000),"lvl":record.levelname,"msg":record.getMessage(),"name":record.name})
def setup_logging():
    h = logging.StreamHandler(sys.stdout); h.setFormatter(JsonFormatter())
    root = logging.getLogger(); root.setLevel(logging.INFO); root.handlers = [h]
