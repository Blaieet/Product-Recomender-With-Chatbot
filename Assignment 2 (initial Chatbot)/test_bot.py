import asyncio as aio

import telepot.aio.api

def _methodurl(req, **user_kw):
    token, method, params, files = req
    return 'http://localhost:9000/bot%s/%s' % (token, method)

telepot.aio.api._methodurl = _methodurl

import telepot
from telepot.aio.loop import MessageLoop
from telepot.aio.delegate import pave_event_space, per_chat_id, \
    create_open, include_callback_query_chat_id


class ShoppingBot(object):
    def __init__(self):
        self.bot = None
        self.loop = aio.get_event_loop()
        
    def start(self, token):
        self.bot = telepot.aio.DelegatorBot(token, [
            include_callback_query_chat_id(
                pave_event_space())(
                per_chat_id(), create_open, ShoppingUser, timeout=10),
            ])
        
        self.loop.create_task(MessageLoop(self.bot).run_forever())
        self.loop.run_forever()

class ShoppingUser(telepot.aio.helper.ChatHandler):
    async def on_chat_message(self, msg):
        content_type, chat_type, chat_id = telepot.glance(msg)
        if 'text' in msg:
            print('From {}: {}'.format(self.id, msg['text']))

    async def on_callback_query(self, msg):
        query_id, from_id, query_data = glance(msg, flavor='callback_query')
        print('Callback {}: {}'.format(self.id, query_data))

    async def on_close(self, ex):
        print('Closed {}'.format(self.id))

if __name__ == '__main__':
    bot = ShoppingBot()
    bot.start("XXXXXXXXXXXXXXXXX")
