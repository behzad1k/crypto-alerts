import json

from faiss import loaded

json_files = [
  # 'db1.json',
  # 'db3.json',
  # 'db4.json',
  'db5.json'
]
tables = []

for json_file in json_files:
  with open(f'results/{json_file}') as f:
    loaded_json = json.load(f)
    for indicator, value in loaded_json.items():
      buy_accuracy = value['buy_successful']/value['buy_signals']
      if indicator not in [map(lambda x: x['indicator'], tables)]:
        tables.append({
          'indicator': indicator,
          'accuracy': buy_accuracy,
          'success': value['buy_successful'],
          'total': value['buy_signals']
        })
      else:
        indicator_index = tables.index(next((index for (index, d) in enumerate(tables) if "indicator" in d and d["indicator"] == indicator), None))
        successfulSignals = tables[indicator_index]['success'] + value['buy_successful']
        newBuySignals = tables[indicator_index]['total'] + value['buy_signals']
        tables[indicator_index] = {
          'indicator': indicator,
          'success': successfulSignals,
          'total': newBuySignals,
          'accuracy': successfulSignals/newBuySignals
        }

tables = sorted([x for x in tables if x['accuracy'] > .75], key=lambda x: x['accuracy'], reverse=True)
json.dump(tables, open('../patterns.json', 'w'))
# for x in tables[:100]:
  # print(f"{x['indicator']}: {x['accuracy']*100:.2f}% / {x['total']}")