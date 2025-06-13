function getAllSplits(str) {
  const results = [];

  function splitString(prefix, remaining) {
    if (remaining.length === 0) {
      results.push(prefix);
    } else {
      for (let i = 1; i <= remaining.length; i++) {
        const newPrefix = prefix ? `${prefix} ${remaining.slice(0, i)}` : remaining.slice(0, i);
        splitString(newPrefix, remaining.slice(i));
      }
    }
  }

  splitString('', str);
  return results;
}

const inputString = "经常有意见分歧";
const splits = getAllSplits(inputString);

console.log(splits);


