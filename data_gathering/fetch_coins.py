import requests

nobitex_response = requests.get('https://apiv2.nobitex.ir/market/stats')
nobitex_json = nobitex_response.json()
coins = []
for key, val in nobitex_json['stats'].items():
  if 'dayChange' in val:
    coins.append({'symbol': key, 'dayChange': val['dayChange']})

sorted_coins = sorted([x for x in coins if x['symbol'].find('usdt') and float(x['dayChange']) > 0], key=lambda c: c['dayChange'], reverse=True)

print(list(map(lambda c: c['symbol'].split('-')[0].upper(), sorted_coins[:100])))