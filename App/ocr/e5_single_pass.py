"""One valid Tesseract observation is enough to stop the E5 OCR loop.
Arithmetic only validates observed values; it never generates a candidate.
"""
from decimal import Decimal


def read_sequential(image, exe, *, row_id, field, read, upscale, parse):
    first = read(image, exe, row_id=row_id, field=field,
                 variant='raw_digits', whitelist=True)
    if not first.get('error') and parse(first.get('text')) is not None:
        return [first]
    second = read(upscale(image), exe, row_id=row_id, field=field,
                  variant='3x_plain', whitelist=False)
    return [first, second]


def valid_observation(attempts, parse):
    return next((a for a in attempts if not a.get('error') and
                 parse(a.get('text')) is not None), None)


def math_confirms(rows, row_id, field, text, *, printed_total, footer_attempts,
                  vat_rate, parse, min_footer_conf=.86):
    """Conservative optional promotion before E4; otherwise return candidate only."""
    total = parse(printed_total)
    if total is None or total <= 0 or not rows:
        return False
    trusted = set()
    for attempt in footer_attempts or ():
        v = parse(attempt.get('total'))
        score = attempt.get('confidence')
        if score is None and attempt.get('lines'):
            score = min((line.get('confidence') or 0 for line in attempt['lines']), default=0)
        try:
            confident = float(score) >= min_footer_conf
        except (ValueError, TypeError):
            confident = False
        if confident and v is not None:
            trusted.add(v)
    if trusted != {total}:
        return False
    computed = Decimal(0)
    products = Decimal(0)
    taxes = Decimal(0)
    tax_count = 0
    found = False
    for row in rows:
        cells = row.get('cells', {})
        nums = {}
        for f in ('quantity', 'price', 'printed_total'):
            if f == 'printed_total' and f not in cells:
                continue
            if f not in cells:
                return False
            target = row.get('row_id') == row_id and f == field
            value = parse(text if target else cells[f].get('text'))
            if value is None or value < 0:
                return False
            found |= target
            nums[f] = value
        line = nums['quantity'] * nums['price']
        if 'printed_total' in nums and abs(nums['printed_total'] - line) > Decimal('.01'):
            return False
        computed += line
        if row.get('kind') == 'tax':
            tax_count += 1
            taxes += line
        elif row.get('kind') == 'product':
            products += line
        else:
            return False
    if tax_count:
        rate = parse(vat_rate)
        if tax_count != 1 or rate is None or rate < 0 or abs(taxes-products*rate/100) > Decimal('1'):
            return False
    return found and abs(computed-total) <= Decimal('.01')
