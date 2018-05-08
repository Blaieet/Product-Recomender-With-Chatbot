import asyncio as aio
import re
import json

import telepot
from telepot.aio.loop import MessageLoop
from telepot.aio.delegate import pave_event_space, per_chat_id, \
    create_open, include_callback_query_chat_id
    
    
ADDING, FLAGGING = 0, 1
PENDING, BOUGHT = 0, 1
users = {}

def user_info(user_id):
    try: return users[user_id]
    except:
        users[user_id] = {
            'id': user_id, 
            'status': ADDING,
            'messages': [],
            'products': {}
        }
        
        return users[user_id]

def add_product(user_id, prod_id, qty):
    try:
        user_info(user_id)['products'][prod_id]['qty'] += qty
    except:
        user_info(user_id)['products'][prod_id] = {'status': PENDING, 'qty': qty}
        
def clear_info(user_id):
    del users[user_id]
    

class ShoppingBot(object):    
    instance = None
    
    def __init__(self, verbose=False):
        assert ShoppingBot.instance is None
        ShoppingBot.instance = self
        
        self.verbose = verbose
        self.bot = None
        self.loop = aio.get_event_loop()
        self.msg_loop = None
        self.callbacks = {}
        
    def start(self, token):
        self.bot = telepot.aio.DelegatorBot(token, [
            include_callback_query_chat_id(
                pave_event_space())(
                per_chat_id(), create_open, ShoppingUser, timeout=10),
            ])
        
        self.msg_loop = MessageLoop(self.bot)
        self.loop.create_task(self.msg_loop.run_forever())
        self.loop.run_forever()
        
    def stop(self):
        ShoppingBot.instance = None
        if self.msg_loop:
            self.msg_loop.cancel()
            self.msg_loop = None
    
    @classmethod
    def is_verbose(cls):
        return cls.instance.verbose
    
    @classmethod
    def add_callback(cls, event, cb):
        assert event in ('cmd', 'add-product', 'flag-product', 'end')
        
        cls.instance.callbacks[event] = cls.instance.callbacks.get(event, [])
        cls.instance.callbacks[event].append(cb)
        
    @classmethod
    async def trigger(cls, event, user, *args, **kwargs):
        assert event in ('cmd', 'add-product', 'flag-product', 'end')
        
        result = True
        for cb in cls.instance.callbacks.get(event, []):
            cur = await cb(cls.instance.bot, user, *args, **kwargs)
            result = result & (cur is None or cur)
            if result is False:
                return False
            
        return result
        
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, tb):
        self.stop()
        
        
def has_unbought_products(user):
    return any(p['status'] == PENDING for _,p in user['products'].items())

def get_list(user):
    if has_unbought_products(user):
        return '\n'.join(
            map(
                lambda p: p[0] + ': ' + str(p[1]['qty']), 
                filter(
                    lambda p: p[1]['status'] == PENDING, 
                    user['products'].items()
                )
            )
        )
    
    return 'No hi ha productes'


class ShoppingUser(telepot.aio.helper.ChatHandler):    
    def __init__(self, *args, **kwargs):
        super(ShoppingUser, self).__init__(*args, **kwargs)
        
        if ShoppingBot.is_verbose():
            print('Created {}'.format(self.id))
    
    def is_adding(self):
        return user_info(self.id)['status'] == ADDING
    
    async def on_chat_message(self, msg):
        content_type, chat_type, chat_id = telepot.glance(msg)
        if 'text' in msg:
            msg = msg['text']
            user = user_info(self.id)
                
            if msg == '/done':
                user['status'] = FLAGGING
            
            if msg == '/debug':
                await self.sender.sendMessage(json.dumps(user))
                
            elif msg == '/start':
                clear_info(self.id)
                
            elif msg == '/done' or msg == '/list':
                ans = get_list(user)
                await self.sender.sendMessage(ans)
            elif msg[0] == '/':
                # Special case we don't handle, avoid treating it as a product
                pass
            elif user['status'] == ADDING:
                match = re.findall('^([\w\s]+)\s+([0-9]+)$', msg)
                if len(match) == 1 and len(match[0]) == 2: 
                    item, qty = match[0]
                else:
                    match = re.findall('^([0-9]+)\s+([\w\s]+)$', msg)
                    if len(match) == 1 and len(match[0]) == 2: 
                        qty, item = match[0]
                    else:
                        item, qty = msg, 1

                if await ShoppingBot.trigger('add-product', self, item, int(qty)):
                    add_product(self.id, item, int(qty))
                else:
                    await self.sender.sendMessage('Aquest producte no existeix')
                    
                
            elif user['status'] == FLAGGING:
                try:
                    assert user['products'][msg]['status'] == PENDING
                    
                    if await ShoppingBot.trigger('flag-product', self, msg):
                        user['products'][msg]['status'] = BOUGHT
                        ans = get_list(user)

                        if not has_unbought_products(user):
                            await ShoppingBot.trigger('end', self)
                    
                    await self.sender.sendMessage(ans)
                except KeyError:
                    await self.sender.sendMessage('No tens aquest producte a la llista')
                except:
                    await self.sender.sendMessage('Ja està comprat')
                
            user['messages'].append(msg)
            
            if msg[0] == '/':
                await ShoppingBot.trigger('cmd', self, msg)
                
            if ShoppingBot.is_verbose():
                print('From {}: {}'.format(self.id, msg))

    async def on_close(self, ex):
        if ShoppingBot.is_verbose():
            print('Closed {}'.format(self.id))
