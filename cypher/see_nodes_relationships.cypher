//MATCH (n)-[r]->(m) RETURN m, n, r

// MATCH (n:Document{title: 'KG-paper-google.pdf'})-[r]->(m) 
// RETURN n,r,m

//MATCH (n)-[r]->(m) RETURN m, n, r

MATCH (n:Document)-[r]->(m) 
RETURN n,r,m



// MATCH (n:Document{title: 'FUZZ_IEEE_Paper_version_18-01-2019.pdf'})-[r]->(m) 
// RETURN n,r,m
